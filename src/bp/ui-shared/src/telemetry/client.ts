import { AxiosInstance } from 'axios'
import { buildSchema, ServerStats, TelemetryEvent, TelemetryEventData } from 'common/telemetry'
import _ from 'lodash'
import ms from 'ms'

import { DataCollector, uiLanguageCollector } from './collectors'
import { sendTelemetry } from './index'

interface UIEventsCollector {
  refreshInterval: number
  collectData: DataCollector
}

const eventCollectorStore: _.Dictionary<UIEventsCollector> = {}

const getEventExpiry = (eventName: string) => {
  const expiryAsString = window.localStorage.getItem(eventName)
  return expiryAsString ? ms(expiryAsString) : null
}

const setEventExpiry = (eventName: string) => {
  const expiry = Date.now() + eventCollectorStore[eventName].refreshInterval
  window.localStorage.setItem(eventName, expiry.toString())
}

export const addEventCollector = (eventName: string, interval: string, collector: DataCollector) => {
  eventCollectorStore[eventName] = {
    refreshInterval: ms(interval),
    collectData: collector
  }
}

const sendEventIfReady = async (api: AxiosInstance, eventName: string): Promise<boolean> => {
  const expiry = getEventExpiry(eventName)

  if (expiry && expiry > Date.now()) {
    return false
  }

  const eventData = eventCollectorStore[eventName].collectData()
  const event = await makeTelemetryEvent(api, eventName, eventData)

  return sendTelemetry([event])
}

const expiryEvent = response => {
  for (const eventName in response) {
    response[eventName] && setEventExpiry(eventName)
  }
}

export const startTelemetry = (api: AxiosInstance) => {
  addEventCollector('ui_language', '8h', uiLanguageCollector)

  const interval = setInterval(async () => {
    if (!window.TELEMETRY_URL) {
      return
    }

    const answer = {}
    for (const eventName in eventCollectorStore) {
      const response = await sendEventIfReady(api, eventName)
      answer[eventName] = response
    }

    expiryEvent(answer)

    clearInterval(interval)
  }, ms('1s'))
}

const makeTelemetryEvent = async (
  api: AxiosInstance,
  event_type: string,
  event_data: TelemetryEventData
): Promise<TelemetryEvent> => {
  const serverStats: ServerStats = await api.get('/telemetry/server-config')

  return {
    ...buildSchema(serverStats, 'client'),
    event_type,
    event_data
  }
}
