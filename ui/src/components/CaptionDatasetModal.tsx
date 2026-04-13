import React, { useState, useEffect, useRef } from 'react';
import { Modal } from '@/components/Modal';
import { createGlobalState } from 'react-global-hooks';
import { useFromNull } from '@/hooks/useFromNull';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  NumberInput,
  SelectInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import { CaptionJobConfig } from '@/types';
import { defaultCaptionJobConfig, handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
import { loadCaptionDraft, saveCaptionDraft } from '@/helpers/captionDraftStorage';
import {
  buildCaptionPrompt,
  captionPromptTemplateOptions,
  captionTargetLanguageOptions,
  defaultCaptionPromptTemplate,
  defaultCaptionTargetLanguage,
} from '@/helpers/captionPrompts';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import {
  captionerTypes,
  defaultQtype,
  groupedCaptionerTypes,
  maxNewTokensOptions,
  maxResOptions,
  quantizationOptions,
} from '@/helpers/captionOptions';
import { isMac } from '@/helpers/basic';
import useGPUInfo from '@/hooks/useGPUInfo';
import { apiClient } from '@/utils/api';
import { v4 as uuidv4 } from 'uuid';
import { startJob } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';

export interface CaptionDatasetModalState {
  datasetPath: string;
  onClose?: () => void;
}

export const captionDatasetModalState = createGlobalState<CaptionDatasetModalState | null>(null);

export const openCaptionDatasetModal = (datasetPath: string, onClose?: () => void) => {
  captionDatasetModalState.set({ datasetPath, onClose });
};

export const CaptionDatasetModal: React.FC = () => {
  const [modalInfo, setModalInfo] = captionDatasetModalState.use();
  const [jobConfig, setJobConfig] = useNestedState<CaptionJobConfig>(objectCopy(defaultCaptionJobConfig));
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [isDraftLoaded, setIsDraftLoaded] = useState(false);
  const [apiTestStatus, setApiTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [apiTestMessage, setApiTestMessage] = useState('');
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const open = modalInfo !== null;
  const isSavingRef = useRef(false);
  const isSavingDraftRef = useRef(false);
  const showGPUSelect = !isMac();
  const selectedCaptionOption = captionerTypes.find(option => option.name === jobConfig.config.process[0].type);
  const isRemoteApiCaptioner = jobConfig.config.process[0].type === 'RemoteAPICaptioner';
  const promptTemplateValue =
    jobConfig.config.process[0].caption.prompt_template || defaultCaptionPromptTemplate;
  const targetLanguageValue =
    jobConfig.config.process[0].caption.target_lang || defaultCaptionTargetLanguage;

  useFromNull(() => {
    let isCancelled = false;

    const initializeDraft = async () => {
      const loadedDraft = await loadCaptionDraft(defaultCaptionJobConfig);
      if (isCancelled) {
        return;
      }

      const nextJobConfig = loadedDraft?.jobConfig || objectCopy(defaultCaptionJobConfig);
      if (modalInfo?.datasetPath) {
        nextJobConfig.config.process[0].caption.path_to_caption = modalInfo.datasetPath;
      }

      setJobConfig(nextJobConfig);
      setGpuIDs(loadedDraft?.gpuIDs ?? null);
      setIsDraftLoaded(true);
    };

    setIsDraftLoaded(false);
    initializeDraft();

    return () => {
      isCancelled = true;
    };
  }, [modalInfo]);

  useEffect(() => {
    if (!open || !isDraftLoaded || isSavingDraftRef.current) {
      return;
    }

    const timeout = window.setTimeout(async () => {
      try {
        isSavingDraftRef.current = true;
        await saveCaptionDraft({
          jobConfig,
          gpuIDs,
        });
      } catch (error) {
        console.warn('Failed to save caption draft to server:', error);
      } finally {
        isSavingDraftRef.current = false;
      }
    }, 400);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [jobConfig, gpuIDs, open, isDraftLoaded]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  useEffect(() => {
    if (!selectedCaptionOption?.additionalSections?.includes('caption.caption_prompt')) {
      return;
    }

    const promptTemplate = jobConfig.config.process[0].caption.prompt_template;
    const targetLang = jobConfig.config.process[0].caption.target_lang;
    const captionPrompt = jobConfig.config.process[0].caption.caption_prompt;
    const fallbackPrompt = buildCaptionPrompt(promptTemplate, targetLang);

    if (!promptTemplate) {
      setJobConfig(defaultCaptionPromptTemplate, 'config.process[0].caption.prompt_template');
    }
    if (!targetLang) {
      setJobConfig(defaultCaptionTargetLanguage, 'config.process[0].caption.target_lang');
    }
    if (!captionPrompt) {
      setJobConfig(fallbackPrompt, 'config.process[0].caption.caption_prompt');
    }
  }, [
    selectedCaptionOption,
    jobConfig.config.process[0].caption.prompt_template,
    jobConfig.config.process[0].caption.target_lang,
    jobConfig.config.process[0].caption.caption_prompt,
    setJobConfig,
  ]);

  useEffect(() => {
    if (apiTestStatus !== 'idle') {
      setApiTestStatus('idle');
      setApiTestMessage('');
    }
  }, [
    jobConfig.config.process[0].caption.api_base_url,
    jobConfig.config.process[0].caption.api_concurrency,
    jobConfig.config.process[0].caption.api_key,
    jobConfig.config.process[0].caption.api_protocol,
    jobConfig.config.process[0].caption.model_name_or_path,
    jobConfig.config.process[0].type,
  ]);

  useEffect(() => {
    if (!selectedCaptionOption?.additionalSections?.includes('caption.api_concurrency')) {
      return;
    }

    if (!jobConfig.config.process[0].caption.api_concurrency) {
      setJobConfig(20, 'config.process[0].caption.api_concurrency');
    }
  }, [
    selectedCaptionOption,
    jobConfig.config.process[0].caption.api_concurrency,
    setJobConfig,
  ]);

  const handleClose = () => {
    if (modalInfo?.onClose) {
      modalInfo.onClose();
    }
    setModalInfo(null);
  };

  const applyPromptPreset = (nextTemplate: string, nextTargetLanguage: string) => {
    setJobConfig(nextTemplate, 'config.process[0].caption.prompt_template');
    setJobConfig(nextTargetLanguage, 'config.process[0].caption.target_lang');
    setJobConfig(buildCaptionPrompt(nextTemplate, nextTargetLanguage), 'config.process[0].caption.caption_prompt');
  };

  const testRemoteApi = async () => {
    const captionConfig = jobConfig.config.process[0].caption;
    if (!captionConfig.model_name_or_path?.trim()) {
      setApiTestStatus('error');
      setApiTestMessage('请先填写模型名称。');
      return;
    }
    if (!captionConfig.api_base_url?.trim()) {
      setApiTestStatus('error');
      setApiTestMessage('请先填写 Base URL。');
      return;
    }
    if (!captionConfig.api_key?.trim()) {
      setApiTestStatus('error');
      setApiTestMessage('请先填写 API Key。');
      return;
    }

    setApiTestStatus('testing');
    setApiTestMessage('正在测试 API 连通性...');

    try {
      const response = await apiClient.post('/api/caption/test-api', {
        model_name_or_path: captionConfig.model_name_or_path,
        api_base_url: captionConfig.api_base_url,
        api_key: captionConfig.api_key,
        api_protocol: captionConfig.api_protocol || 'openai',
      });
      const preview = response.data?.preview ? `，返回：${response.data.preview}` : '';
      setApiTestStatus('success');
      setApiTestMessage(`${response.data?.message || 'API 连通性测试通过'}${preview}`);
    } catch (error: any) {
      const details = error?.response?.data?.details || error?.response?.data?.error || error?.message || '未知错误';
      setApiTestStatus('error');
      setApiTestMessage(`测试失败：${details}`);
    }
  };

  const saveJob = async () => {
    if (isSavingRef.current) return;
    if (!modalInfo?.datasetPath) {
      alert('数据集路径缺失，请重试。');
      return;
    }
    isSavingRef.current = true;

    apiClient
      .post('/api/jobs', {
        id: null,
        name: uuidv4(),
        gpu_ids: gpuIDs,
        job_config: jobConfig,
        job_type: 'caption',
        job_ref: modalInfo.datasetPath,
      })
      .then(async res => {
        const jobId = res.data.id;
        await startJob(jobId);
        // start the queue as well
        await startQueue(gpuIDs || '');
        isSavingRef.current = false;
        handleClose();
      })
      .catch(error => {
        if (error.response?.status === 409) {
          alert('该数据集的打标任务已存在，请到任务列表中查看。');
        } else {
          alert('保存任务失败，请重试。');
        }
        console.log('Error saving training:', error);
        isSavingRef.current = false;
      });
  };

  return (
    <Modal isOpen={open} onClose={handleClose} title="数据集打标" size="lg">
      <div className="space-y-4 text-gray-200">
        <form
          onSubmit={e => {
            e.preventDefault();
            saveJob();
          }}
        >
          <div className="text-sm text-gray-400">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div>
                <SelectInput
                  label="打标器类型"
                  value={jobConfig.config.process[0].type}
                  onChange={value => {
                    handleCaptionerTypeChange(jobConfig.config.process[0].type, value, jobConfig, setJobConfig);
                  }}
                  options={groupedCaptionerTypes}
                />
              </div>
              {showGPUSelect && (
                <div>
                  <SelectInput
                    label="GPU 编号"
                    value={`${gpuIDs}`}
                    onChange={value => setGpuIDs(value)}
                    options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
                  />
                </div>
              )}
            </div>
            <div className="mt-4">
              <CreatableSelectInput
                label={isRemoteApiCaptioner ? '模型名称' : '模型名称或路径'}
                value={jobConfig.config.process[0].caption.model_name_or_path}
                docKey="config.process[0].caption.model_name_or_path"
                onChange={(value: string | null) => {
                  if (value?.trim() === '') {
                    value = null;
                  }
                  setJobConfig(value, 'config.process[0].caption.model_name_or_path');
                }}
                placeholder=""
                options={selectedCaptionOption?.name_or_path_options || []}
                required
              />
            </div>
            {selectedCaptionOption?.additionalSections?.includes('caption.api_base_url') && (
              <div className="mt-4">
                <TextInput
                  label="Base URL"
                  value={jobConfig.config.process[0].caption.api_base_url || ''}
                  onChange={value => setJobConfig(value, 'config.process[0].caption.api_base_url')}
                  placeholder={
                    (jobConfig.config.process[0].caption.api_protocol || 'openai') === 'anthropic'
                      ? '例如：https://api.anthropic.com/v1'
                      : '例如：https://api.openai.com/v1'
                  }
                  required
                />
              </div>
            )}
            {selectedCaptionOption?.additionalSections?.includes('caption.api_key') && (
              <div className="mt-4">
                <TextInput
                  label="API Key"
                  type="password"
                  value={jobConfig.config.process[0].caption.api_key || ''}
                  onChange={value => setJobConfig(value, 'config.process[0].caption.api_key')}
                  placeholder="输入用于打标的 API Key"
                  required
                />
              </div>
            )}
            {selectedCaptionOption?.additionalSections?.includes('caption.api_protocol') && (
              <div className="mt-4">
                <label className="block text-xs mb-1 mt-2 text-gray-300">协议标准</label>
                <div className="flex items-center gap-3">
                  <span
                    className={`text-sm ${
                      (jobConfig.config.process[0].caption.api_protocol || 'openai') === 'openai'
                        ? 'text-white font-medium'
                        : 'text-gray-400'
                    }`}
                  >
                    OpenAI
                  </span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={(jobConfig.config.process[0].caption.api_protocol || 'openai') === 'anthropic'}
                    onClick={() =>
                      setJobConfig(
                        (jobConfig.config.process[0].caption.api_protocol || 'openai') === 'anthropic'
                          ? 'openai'
                          : 'anthropic',
                        'config.process[0].caption.api_protocol',
                      )
                    }
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 ${
                      (jobConfig.config.process[0].caption.api_protocol || 'openai') === 'anthropic'
                        ? 'bg-blue-500'
                        : 'bg-gray-600'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        (jobConfig.config.process[0].caption.api_protocol || 'openai') === 'anthropic'
                          ? 'translate-x-5'
                          : 'translate-x-0'
                      }`}
                    />
                  </button>
                  <span
                    className={`text-sm ${
                      (jobConfig.config.process[0].caption.api_protocol || 'openai') === 'anthropic'
                        ? 'text-white font-medium'
                        : 'text-gray-400'
                    }`}
                  >
                    Anthropic
                  </span>
                </div>
              </div>
            )}
            {selectedCaptionOption?.additionalSections?.includes('caption.model_name_or_path2') && (
              <div className="mt-4">
                <CreatableSelectInput
                  label="模型名称或路径 2"
                  value={jobConfig.config.process[0].caption.model_name_or_path2 || ''}
                  onChange={(value: string | null) => {
                    if (value?.trim() === '') {
                      value = null;
                    }
                    setJobConfig(value, 'config.process[0].caption.model_name_or_path2');
                  }}
                  placeholder=""
                  options={selectedCaptionOption?.name_or_path2_options || []}
                />
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div>
                {selectedCaptionOption?.supportsQuantization !== false && (
                  <SelectInput
                    label="量化"
                    value={jobConfig.config.process[0].caption.quantize ? jobConfig.config.process[0].caption.qtype : ''}
                    onChange={value => {
                      if (value === '') {
                        setJobConfig(false, 'config.process[0].caption.quantize');
                        value = defaultQtype;
                      } else {
                        setJobConfig(true, 'config.process[0].caption.quantize');
                      }
                      setJobConfig(value, 'config.process[0].caption.qtype');
                    }}
                    options={quantizationOptions}
                  />
                )}
                {selectedCaptionOption?.additionalSections?.includes('caption.max_res') && (
                  <div className={selectedCaptionOption?.supportsQuantization !== false ? 'mt-4' : ''}>
                    <SelectInput
                      label="最大分辨率"
                      value={`${jobConfig.config.process[0].caption.max_res || ''}`}
                      onChange={value => {
                        const intVal = parseInt(value);
                        if (!isNaN(intVal)) {
                          setJobConfig(intVal, 'config.process[0].caption.max_res');
                        }
                      }}
                      options={maxResOptions}
                    />
                  </div>
                )}
                {selectedCaptionOption?.additionalSections?.includes('caption.max_new_tokens') && (
                  <div className="mt-4">
                    <SelectInput
                      label="最大新 Token 数"
                      value={`${jobConfig.config.process[0].caption.max_new_tokens || ''}`}
                      onChange={value => {
                        const intVal = parseInt(value);
                        if (!isNaN(intVal)) {
                          setJobConfig(intVal, 'config.process[0].caption.max_new_tokens');
                        }
                      }}
                      options={maxNewTokensOptions}
                    />
                  </div>
                )}
                {selectedCaptionOption?.additionalSections?.includes('caption.api_concurrency') && (
                  <div className="mt-4">
                    <NumberInput
                      label="并发数"
                      value={jobConfig.config.process[0].caption.api_concurrency || 20}
                      onChange={value => {
                        const safeValue = value === null ? 20 : Math.max(1, Math.min(100, Math.floor(value)));
                        setJobConfig(safeValue, 'config.process[0].caption.api_concurrency');
                      }}
                      min={1}
                      max={100}
                      placeholder="默认 20"
                    />
                  </div>
                )}
                {selectedCaptionOption?.additionalSections?.includes('caption.prompt_template') && (
                  <div className="mt-4">
                    <SelectInput
                      label="打标模式"
                      value={promptTemplateValue}
                      onChange={value => applyPromptPreset(value, targetLanguageValue)}
                      options={captionPromptTemplateOptions}
                    />
                  </div>
                )}
                {selectedCaptionOption?.additionalSections?.includes('caption.target_lang') && (
                  <div className="mt-4">
                    <SelectInput
                      label="打标语言"
                      value={targetLanguageValue}
                      onChange={value => applyPromptPreset(promptTemplateValue, value)}
                      options={captionTargetLanguageOptions}
                    />
                  </div>
                )}
              </div>
              <div>
                <FormGroup label="选项">
                  {isRemoteApiCaptioner && (
                    <div className="mb-4">
                      <button
                        type="button"
                        onClick={testRemoteApi}
                        disabled={apiTestStatus === 'testing'}
                        className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                          apiTestStatus === 'testing'
                            ? 'bg-gray-600 text-gray-300 cursor-not-allowed'
                            : 'bg-sky-600 text-white hover:bg-sky-700'
                        }`}
                      >
                        {apiTestStatus === 'testing' ? '测试中...' : '测试 API'}
                      </button>
                      {apiTestStatus !== 'idle' && (
                        <p
                          className={`mt-2 text-sm break-words ${
                            apiTestStatus === 'success' ? 'text-green-400' : apiTestStatus === 'error' ? 'text-red-400' : 'text-gray-300'
                          }`}
                        >
                          {apiTestMessage}
                        </p>
                      )}
                    </div>
                  )}
                  {selectedCaptionOption?.supportsLowVram !== false && (
                    <Checkbox
                      label="低显存"
                      checked={jobConfig.config.process[0].caption.low_vram}
                      onChange={value => setJobConfig(value, 'config.process[0].caption.low_vram')}
                    />
                  )}
                  <Checkbox
                    label="重新打标"
                    checked={jobConfig.config.process[0].caption.recaption}
                    onChange={value => setJobConfig(value, 'config.process[0].caption.recaption')}
                  />
                </FormGroup>
              </div>
            </div>
            {selectedCaptionOption?.additionalSections?.includes('caption.caption_prompt') && (
              <div className="mt-4">
                <TextAreaInput
                  label="打标提示词"
                  value={jobConfig.config.process[0].caption.caption_prompt || ''}
                  onChange={value => {
                    setJobConfig(value, 'config.process[0].caption.caption_prompt');
                  }}
                  placeholder="输入打标提示词"
                />
              </div>
            )}
          </div>

          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
              onClick={handleClose}
            >
              取消
            </button>
            <button
              type="submit"
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              加入队列
            </button>
          </div>
        </form>
      </div>
    </Modal>
  );
};
