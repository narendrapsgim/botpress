// tslint:disable-next-line:ordered-imports
import '../../worker-before'
// tslint:disable-next-line:ordered-imports
import '../../sdk/rewire'
// tslint:disable-next-line:ordered-imports
import * as sdk from 'botpress/sdk'
import _ from 'lodash'
import { isMainThread, parentPort, workerData } from 'worker_threads'

import { getMinKFold } from './grid-search/split-dataset'
import { SVM } from './svm'
import { Data, KernelTypes, SvmModel, SvmParameters as Parameters, SvmTypes } from './typings'

if (isMainThread) {
  throw new Error('I am not a worker')
}

console.log('Hello from worker')

const points: sdk.MLToolkit.SVM.DataPoint[] = workerData.points
const options: Partial<sdk.MLToolkit.SVM.SVMOptions> = workerData.options

const vectorsLengths = _(points)
  .map(p => p.coordinates.length)
  .uniq()
  .value()
if (vectorsLengths.length > 1) {
  throw new Error('All vectors must be of the same size')
}

const labels = _(points)
  .map(p => p.label)
  .uniq()
  .value()
const dataset: Data[] = points.map(p => [p.coordinates, labels.indexOf(p.label)])

if (labels.length < 2) {
  throw new Error("SVM can't train on a dataset of only one class")
}

const minKFold = getMinKFold(dataset)
const kFold = Math.max(minKFold, 4)

const arr = (n: number | number[]) => (_.isArray(n) ? n : [n])

const svm = new SVM({
  svm_type: options.classifier ? SvmTypes[options.classifier] : undefined,
  kernel_type: options.kernel ? KernelTypes[options.kernel] : undefined,
  C: options.c ? arr(options.c) : undefined,
  gamma: options.gamma ? arr(options.gamma) : undefined,
  probability: options.probability,
  reduce: options.reduce,
  kFold
})

svm
  .train(dataset, progress => {
    parentPort?.postMessage({
      status: 'progress',
      data: progress
    })
  })
  .then(results => {
    parentPort?.postMessage({
      status: 'done',
      data: results ? JSON.stringify({ ...results.model, labels_idx: labels }) : undefined
    })
  })
  .catch(err => {
    parentPort?.postMessage({
      status: 'error',
      data: err
    })
  })
