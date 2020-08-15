import { TelemetryEventData } from 'common/telemetry'
import _ from 'lodash'

import { langLocale } from '../translations'

export type DataCollector = () => TelemetryEventData

export const uiLanguageCollector = (): TelemetryEventData => {
  return {
    schema: '1.0.0',
    user: {
      email, // TODO add email getter
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    },
    language: langLocale()
  }
}
