import * as sdk from 'botpress/sdk'
import _ from 'lodash'

import { Trainer } from './svm'

export class SVMTrainingPool {
  private canceledTrainSessions: string[] = [] // TODO: purge this array once training is done
  private currentSvms: _.Dictionary<Trainer> = {}

  public async startTraining(
    trainingId: string,
    trainSessionId: string,
    points: sdk.MLToolkit.SVM.DataPoint[],
    options: Partial<sdk.MLToolkit.SVM.SVMOptions>,
    progress: sdk.MLToolkit.SVM.TrainProgressCallback | undefined,
    complete: (model: string) => void,
    error: (error: Error) => void
  ) {
    if (!!this.currentSvms[trainingId]) {
      error(new Error('this exact training was already started'))
      return
    }
    if (this.canceledTrainSessions.includes(trainSessionId)) {
      complete('')
    }

    this.currentSvms[trainingId] = new Trainer(trainSessionId)
    try {
      const result = await this.currentSvms[trainingId].train(points, options, progress)
      complete(result)
    } catch (err) {
      error(err)
    } finally {
      delete this.currentSvms[trainingId]
    }
  }

  public cancelAll(trainSessionId: string) {
    if (!this.canceledTrainSessions.includes(trainSessionId)) {
      this.canceledTrainSessions.push(trainSessionId)
    }

    for (const svm of Object.values(this.currentSvms)) {
      if (svm.trainSessionId === trainSessionId) {
        svm.cancelTraining()
        delete this.currentSvms[svm.trainingId]
      }
    }
  }
}
