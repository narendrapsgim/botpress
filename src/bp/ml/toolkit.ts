import * as sdk from 'botpress/sdk'
import cluster, { Worker } from 'cluster'
import _ from 'lodash'
import kmeans from 'ml-kmeans'

import { registerMsgHandler, spawnMLWorkers, WORKER_TYPES } from '../cluster'

import { Tagger, Trainer as CRFTrainer } from './crf'
import { CRFTrainingPool } from './crf-pool'
import { FastTextModel } from './fasttext'
import computeJaroWinklerDistance from './homebrew/jaro-winkler'
import computeLevenshteinDistance from './homebrew/levenshtein'
import { processor } from './sentencepiece'
import { Predictor, Trainer as SVMTrainer } from './svm'
import { SVMTrainingPool } from './svm-pool'

type MsgType =
  | 'svm_train'
  | 'svm_progress'
  | 'svm_done'
  | 'svm_error'
  | 'svm_kill_all'
  | 'crf_train'
  | 'crf_progress'
  | 'crf_done'
  | 'crf_error'
  | 'crf_kill'

interface Message {
  type: MsgType
  trainSessionId: string
  workerPid?: number
}

interface TrainingMessage extends Message {
  trainingId: string
  payload: any
}

// assuming 10 bots, 10 ctx * (oos, intent) + ndu + ctx cls + slot tagger
// all training concurrently
const MAX_TRAINING_LISTENENRS = 10 * (10 * 2 + 2)

const MLToolkit: typeof sdk.MLToolkit = {
  KMeans: {
    kmeans
  },
  CRF: {
    Tagger,
    Trainer: CRFTrainer
  },
  SVM: {
    Predictor,
    Trainer: SVMTrainer
  },
  FastText: { Model: FastTextModel },
  Strings: { computeLevenshteinDistance, computeJaroWinklerDistance },
  SentencePiece: { createProcessor: processor }
}

function overloadTrainers() {
  MLToolkit.SVM.Trainer.prototype.train = function(
    points: sdk.MLToolkit.SVM.DataPoint[],
    options?: Partial<sdk.MLToolkit.SVM.SVMOptions>,
    progressCb?: sdk.MLToolkit.SVM.TrainProgressCallback | undefined
  ): any {
    process.setMaxListeners(MAX_TRAINING_LISTENENRS)

    return Promise.fromCallback(completedCb => {
      const messageHandler = (msg: TrainingMessage) => {
        if (msg.trainingId !== this.trainingId) {
          return
        }
        if (progressCb && msg.type === 'svm_progress') {
          try {
            progressCb(msg.payload.progress)
          } catch (err) {
            completedCb(err)

            const { trainSessionId, workerPid } = msg
            const killMsg: Message = { type: 'svm_kill_all', trainSessionId, workerPid }
            process.send!(killMsg) // kill all of worker's svm

            process.off('message', messageHandler)
          }
        }

        if (msg.type === 'svm_done') {
          completedCb(undefined, msg.payload.result)
          process.off('message', messageHandler)
        }

        if (msg.type === 'svm_error') {
          completedCb(msg.payload.error)
          process.off('message', messageHandler)
        }
      }

      const { trainingId, trainSessionId } = this
      const trainMsg: TrainingMessage = { type: 'svm_train', trainingId, trainSessionId, payload: { points, options } }
      process.send!(trainMsg)
      process.on('message', messageHandler)
    })
  }

  MLToolkit.CRF.Trainer.prototype.train = function(
    elements: sdk.MLToolkit.CRF.DataPoint[],
    params: sdk.MLToolkit.CRF.TrainerOptions,
    progressCb?: (iteration: number) => void
  ): Promise<string> {
    return Promise.fromCallback(completedCb => {
      const messageHandler = (msg: TrainingMessage) => {
        if (msg.trainingId !== this.trainingId) {
          return
        }

        if (progressCb && msg.type === 'crf_progress') {
          try {
            progressCb(msg.payload.progress)
          } catch (err) {
            completedCb(err)

            const { workerPid } = msg
            const killMsg: TrainingMessage = { type: 'crf_kill', trainingId, trainSessionId, payload: {}, workerPid }
            process.send!(killMsg)

            process.off('message', messageHandler)
          }
        }

        if (msg.type === 'crf_done') {
          completedCb(undefined, msg.payload.crfModelFilename)
          process.off('message', messageHandler)
        }

        if (msg.type === 'crf_error') {
          completedCb(msg.payload.error)
          process.off('message', messageHandler)
        }
      }

      const { trainingId, trainSessionId } = this
      const trainMsg: TrainingMessage = { type: 'crf_train', trainingId, trainSessionId, payload: { elements, params } }
      process.send!(trainMsg)
      process.on('message', messageHandler)
    }) as any
  }
}

if (cluster.isWorker) {
  if (process.env.WORKER_TYPE === WORKER_TYPES.WEB) {
    overloadTrainers()
  }
  if (process.env.WORKER_TYPE === WORKER_TYPES.ML) {
    const svmPool = new SVMTrainingPool() // one svm pool per ml worker
    const crfPool = new CRFTrainingPool()
    async function messageHandler(msg: TrainingMessage) {
      if (msg.type === 'svm_train') {
        const { trainingId, trainSessionId, payload } = msg
        const { points, options } = payload
        let svmProgressCalls = 0

        // tslint:disable-next-line: no-floating-promises
        await svmPool.startTraining(
          trainingId,
          trainSessionId,
          points,
          options,
          progress => {
            if (++svmProgressCalls % 10 === 0 || progress === 1) {
              const progressMsg: TrainingMessage = {
                type: 'svm_progress',
                trainingId,
                trainSessionId,
                payload: { progress },
                workerPid: process.pid
              }
              process.send!(progressMsg)
            }
          },
          result => {
            const completedMsg: TrainingMessage = { type: 'svm_done', trainingId, trainSessionId, payload: { result } }
            process.send!(completedMsg)
          },
          error => {
            const errorMsg: TrainingMessage = { type: 'svm_error', trainingId, trainSessionId, payload: { error } }
            process.send!(errorMsg)
          }
        )
      }

      if (msg.type === 'svm_kill_all') {
        svmPool.cancelAll(msg.trainSessionId)
      }

      if (msg.type === 'crf_train') {
        const { trainingId, trainSessionId, payload } = msg
        const { elements, params } = payload
        // tslint:disable-next-line: no-floating-promises
        crfPool.startTraining(
          msg.trainingId,
          msg.trainSessionId,
          elements,
          params,
          iteration => {
            const progressMsg: TrainingMessage = {
              type: 'crf_progress',
              trainingId,
              trainSessionId,
              payload: { iteration },
              workerPid: process.pid
            }
            process.send!(progressMsg)
            return 0
          },
          crfModelFilename => {
            const completedMsg: TrainingMessage = {
              type: 'crf_done',
              trainingId,
              trainSessionId,
              payload: { crfModelFilename }
            }
            process.send!(completedMsg)
          },
          error => {
            const errorMsg: TrainingMessage = { type: 'crf_error', trainingId, trainSessionId, payload: { error } }
            process.send!(errorMsg)
          }
        )
      }

      if (msg.type === 'crf_kill') {
        crfPool.cancelTraining(msg.trainingId)
      }
    }

    process.on('message', messageHandler)
  }
}

if (cluster.isMaster) {
  function sendToWebWorker(msg: Message) {
    const webWorker = cluster.workers[process.WEB_WORKER]
    webWorker?.isConnected() && webWorker.send(msg)
  }

  let spawnPromise: Promise<void> | undefined
  async function pickMLWorker(): Promise<Worker> {
    if (_.isEmpty(process.ML_WORKERS) && !spawnPromise) {
      spawnPromise = spawnMLWorkers()
    }
    if (spawnPromise) {
      await spawnPromise
      spawnPromise = undefined
    }

    const idx = Math.floor(Math.random() * process.ML_WORKERS.length)
    const workerID = process.ML_WORKERS[idx]
    const worker = cluster.workers[workerID!]
    if (worker?.isDead() || !worker?.isConnected()) {
      process.ML_WORKERS.splice(idx, 1)
      return pickMLWorker()
    }

    return worker
  }

  function getMLWorker(pid?: number): Worker | undefined {
    if (!pid) {
      return
    }
    return Object.values(cluster.workers).find(w => w && w.process.pid === pid)
  }

  registerMsgHandler('svm_done', sendToWebWorker)
  registerMsgHandler('svm_progress', sendToWebWorker)
  registerMsgHandler('svm_error', sendToWebWorker)
  registerMsgHandler('svm_train', async (msg: Message) => (await pickMLWorker()).send(msg))
  // TODO: broadcast on all workers instead of killing only svm of current worker
  registerMsgHandler('svm_kill_all', async (msg: Message) => getMLWorker(msg.workerPid)?.send(msg))

  registerMsgHandler('crf_done', sendToWebWorker)
  registerMsgHandler('crf_progress', sendToWebWorker)
  registerMsgHandler('crf_error', sendToWebWorker)
  registerMsgHandler('crf_train', async (msg: Message) => (await pickMLWorker()).send(msg))
  registerMsgHandler('crf_kill', async (msg: TrainingMessage) => getMLWorker(msg.workerPid)?.send(msg))
}

export default MLToolkit
