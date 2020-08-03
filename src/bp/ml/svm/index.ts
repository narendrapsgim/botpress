import * as sdk from 'botpress/sdk'
import { workers } from 'cluster'
import _ from 'lodash'
import path from 'path'
import { isMainThread, Worker, WorkerOptions } from 'worker_threads'

import { getMinKFold } from './grid-search/split-dataset'
import { SVM } from './svm'
import { Data, KernelTypes, SvmModel, SvmParameters as Parameters, SvmTypes } from './typings'

type Serialized = SvmModel & {
  labels_idx: string[]
}

export class Trainer implements sdk.MLToolkit.SVM.Trainer {
  private model?: SvmModel
  private worker?: Worker
  constructor() {
    // process.on('message', message => {
    //   if (message?.type === 'svm_kill') {
    //     this.cancelTraining()
    //   }
    // })
  }

  async cancelTraining() {
    try {
      console.log('Cancelling SVM', this.worker)
      this.worker?.postMessage('SIGINT')
      this.worker?.postMessage('SIGKILL')
      const code = await this.worker?.terminate()
      console.log('TERMINATED = ' + code)
    } finally {
    }
  }

  async train(
    points: sdk.MLToolkit.SVM.DataPoint[],
    options?: Partial<sdk.MLToolkit.SVM.SVMOptions>,
    callback?: sdk.MLToolkit.SVM.TrainProgressCallback | undefined
  ): Promise<string> {
    if (!isMainThread) {
      throw new Error('Can only be called from main thread')
    }
    setTimeout(() => this.cancelTraining(), 5 * 1000)
    return new Promise((resolve, reject) => {
      const clean = data => _.omitBy(data, val => val == undefined || val == undefined || typeof val === 'object')
      const processData = {
        VERBOSITY_LEVEL: process.VERBOSITY_LEVEL,
        IS_PRODUCTION: process.IS_PRODUCTION,
        IS_PRO_AVAILABLE: process.IS_PRO_AVAILABLE,
        BPFS_STORAGE: process.BPFS_STORAGE,
        APP_DATA_PATH: process.APP_DATA_PATH,
        ROOT_PATH: process.ROOT_PATH,
        IS_LICENSED: process.IS_LICENSED,
        IS_PRO_ENABLED: process.IS_PRODUCTION,
        BOTPRESS_VERSION: process.BOTPRESS_VERSION,
        SERVER_ID: process.SERVER_ID,
        LOADED_MODULES: process.LOADED_MODULES,
        PROJECT_LOCATION: process.PROJECT_LOCATION,
        pkg: process.pkg
      }

      const worker = new Worker(path.resolve(__dirname, './worker.js'), ({
        workerData: {
          points,
          options: options ?? {},
          processData: clean(processData),
          processEnv: clean(process.env)
        },
        env: { ...process.env }
      } as any) as WorkerOptions) // TODO: update nodejs typings to Node 12
      this.worker = worker
      worker.on('message', async msg => {
        console.log('Data received from WORKER: ', msg)
        if (msg?.status === 'done') {
          resolve(msg.data)
        }
        if (msg?.status === 'progress') {
          callback?.(msg.data)
        }
        if (msg?.status === 'error') {
          try {
            await worker.terminate()
          } finally {
            reject(msg.data)
          }
        }
      })
      worker.on('exit', code => {
        console.log('Worker Exited' + code)
        if (code > 0) {
          reject(`Exited with code ${code}`)
        }
        delete this.worker
      })
      worker.on('online', msg => {
        console.log('Worker Online' + msg)
      })
      worker.on('error', msg => {
        console.log('Worker Error' + msg)
        reject(msg)
      })
    })

    // const vectorsLengths = _(points)
    //   .map(p => p.coordinates.length)
    //   .uniq()
    //   .value()

    // if (vectorsLengths.length > 1) {
    //   throw new Error('All vectors must be of the same size')
    // }

    // const labels = _(points)
    //   .map(p => p.label)
    //   .uniq()
    //   .value()
    // const dataset: Data[] = points.map(p => [p.coordinates, labels.indexOf(p.label)])

    // if (labels.length < 2) {
    //   throw new Error("SVM can't train on a dataset of only one class")
    // }

    // const minKFold = getMinKFold(dataset)
    // const kFold = Math.max(minKFold, 4)

    // const arr = (n: number | number[]) => (_.isArray(n) ? n : [n])

    // options = options ?? {}
    // this.svm = new SVM({
    //   svm_type: options.classifier ? SvmTypes[options.classifier] : undefined,
    //   kernel_type: options.kernel ? KernelTypes[options.kernel] : undefined,
    //   C: options.c ? arr(options.c) : undefined,
    //   gamma: options.gamma ? arr(options.gamma) : undefined,
    //   probability: options.probability,
    //   reduce: options.reduce,
    //   kFold
    // })

    // const trainResult = await this.svm.train(dataset, progress => {
    //   if (callback && typeof callback === 'function') {
    //     callback(progress)
    //   }
    // })
    // if (!trainResult) {
    //   return ''
    // }

    // const { model } = trainResult
    // this.model = model
    // const serialized: Serialized = { ...model, labels_idx: labels }
    // return JSON.stringify(serialized)
  }

  isTrained(): boolean {
    return !!this.model
  }
}

export class Predictor implements sdk.MLToolkit.SVM.Predictor {
  private clf: SVM | undefined
  private labels: string[]
  private parameters: Parameters | undefined

  constructor(json_model: string) {
    const serialized: Serialized = JSON.parse(json_model)
    this.labels = serialized.labels_idx

    try {
      // TODO: actually check the model format
      const model = _.omit(serialized, 'labels_idx')
      this.parameters = model.param
      this.clf = new SVM({ kFold: 1 }, model)
    } catch (err) {
      this.throwModelHasChanged(err)
    }
  }

  private throwModelHasChanged(err?: Error) {
    let errorMsg = 'SVM model format has changed. NLU needs to be retrained.'
    if (err) {
      errorMsg += ` Inner error is '${err}'.`
    }
    throw new Error(errorMsg)
  }

  private getLabelByIdx(idx): string {
    idx = Math.round(idx)
    if (idx < 0) {
      throw new Error(`Invalid prediction, prediction must be between 0 and ${this.labels.length}`)
    }

    return this.labels[idx]
  }

  async predict(coordinates: number[]): Promise<sdk.MLToolkit.SVM.Prediction[]> {
    if (this.parameters?.probability) {
      return this._predictProb(coordinates)
    } else {
      return await this._predictOne(coordinates)
    }
  }

  private async _predictProb(coordinates: number[]): Promise<sdk.MLToolkit.SVM.Prediction[]> {
    const results = await (this.clf as SVM).predictProbabilities(coordinates)
    const reducedResults = _.reduce(
      Object.keys(results),
      (acc, curr) => {
        const label = this.getLabelByIdx(curr).replace(/__k__\d+$/, '')
        acc[label] = (acc[label] || 0) + results[curr]
        return acc
      },
      {}
    )

    return _.orderBy(
      Object.keys(reducedResults).map(idx => ({ label: idx, confidence: reducedResults[idx] })),
      'confidence',
      'desc'
    )
  }

  private async _predictOne(coordinates: number[]): Promise<sdk.MLToolkit.SVM.Prediction[]> {
    // might simply use oneclass instead
    const results = await (this.clf as SVM).predict(coordinates)
    return [
      {
        label: this.getLabelByIdx(results),
        confidence: 0
      }
    ]
  }

  isLoaded(): boolean {
    return !!this.clf
  }

  getLabels(): string[] {
    return _.values(this.labels)
  }
}
