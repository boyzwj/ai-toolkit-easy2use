import { CaptionJobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { apiClient } from '@/utils/api';

export interface CaptionDraftState {
  jobConfig: CaptionJobConfig;
  gpuIDs: string | null;
}

const isPlainObject = (value: unknown): value is Record<string, any> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const deepMerge = <T>(base: T, incoming: unknown): T => {
  if (Array.isArray(base)) {
    return (Array.isArray(incoming) ? incoming : base) as T;
  }

  if (isPlainObject(base)) {
    const result: Record<string, any> = { ...base };
    if (!isPlainObject(incoming)) {
      return result as T;
    }
    for (const key of Object.keys(incoming)) {
      const baseValue = result[key];
      const incomingValue = incoming[key];
      if (baseValue === undefined) {
        result[key] = incomingValue;
      } else {
        result[key] = deepMerge(baseValue, incomingValue);
      }
    }
    return result as T;
  }

  return (incoming !== undefined ? incoming : base) as T;
};

export const loadCaptionDraft = async (defaultJobConfig: CaptionJobConfig): Promise<CaptionDraftState | null> => {
  try {
    const response = await apiClient.get('/api/settings/caption-draft');
    const parsed = response.data?.draft as Partial<CaptionDraftState> | null | undefined;
    if (!parsed) {
      return null;
    }
    const mergedJobConfig = deepMerge(objectCopy(defaultJobConfig), parsed.jobConfig);
    return {
      jobConfig: mergedJobConfig,
      gpuIDs: parsed.gpuIDs ?? null,
    };
  } catch (error) {
    console.warn('Failed to load caption draft from server:', error);
    return null;
  }
};

export const saveCaptionDraft = async (draft: CaptionDraftState) => {
  await apiClient.post('/api/settings/caption-draft', draft);
};
