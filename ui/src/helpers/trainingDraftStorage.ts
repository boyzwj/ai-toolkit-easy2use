import { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { apiClient } from '@/utils/api';

export interface TrainingDraftState {
  jobConfig: JobConfig;
  gpuIDs: string | null;
}

const LS_KEY = 'aitk_training_draft';

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

// load draft: sync from localStorage (instant), then async from server (background update)
export const loadTrainingDraft = async (defaultJobConfig: JobConfig): Promise<TrainingDraftState | null> => {
  let draft: TrainingDraftState | null = null;

  // 1. Try localStorage (no race condition on fast navigation)
  try {
    const raw = typeof window !== 'undefined' ? localStorage.getItem(LS_KEY) : null;
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<TrainingDraftState>;
      if (parsed?.jobConfig) {
        draft = {
          jobConfig: deepMerge(objectCopy(defaultJobConfig), parsed.jobConfig),
          gpuIDs: parsed.gpuIDs ?? null,
        };
      }
    }
  } catch (e) {
    console.warn('[Draft] localStorage read failed:', e);
  }

  // 2. Try server (background update, may arrive later)
  try {
    const response = await apiClient.get('/api/settings/training-draft');
    const parsed = response.data?.draft as Partial<TrainingDraftState> | null | undefined;
    if (parsed?.jobConfig) {
      draft = {
        jobConfig: deepMerge(objectCopy(defaultJobConfig), parsed.jobConfig),
        gpuIDs: parsed.gpuIDs ?? null,
      };
    }
  } catch (error) {
    // server is optional fallback
  }

  return draft;
};

// save draft: sync to localStorage immediately, then async to server
export const saveTrainingDraft = async (draft: TrainingDraftState) => {
  // localStorage: synchronous, always succeeds even during navigation
  try {
    if (typeof window !== 'undefined') {
      localStorage.setItem(LS_KEY, JSON.stringify(draft));
    }
  } catch (e) {
    console.warn('[Draft] localStorage write failed:', e);
  }

  // server: async, fire-and-forget
  apiClient.post('/api/settings/training-draft', draft).catch(e =>
    console.warn('[Draft] server save failed:', e),
  );
};
