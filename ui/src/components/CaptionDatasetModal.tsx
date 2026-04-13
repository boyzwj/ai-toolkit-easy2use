import React, { useState, useEffect, useRef } from 'react';
import { Modal } from '@/components/Modal';
import { createGlobalState } from 'react-global-hooks';
import { useFromNull } from '@/hooks/useFromNull';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  SelectInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import { CaptionJobConfig } from '@/types';
import { defaultCaptionJobConfig, handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
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
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const open = modalInfo !== null;
  const isSavingRef = useRef(false);
  const showGPUSelect = !isMac();

  useFromNull(() => {
    // reset the state
    setJobConfig(objectCopy(defaultCaptionJobConfig));
    // set the path_to_caption
    if (modalInfo?.datasetPath) {
      setJobConfig(modalInfo.datasetPath, 'config.process[0].caption.path_to_caption');
    }
  }, [modalInfo]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  const handleClose = () => {
    if (modalInfo?.onClose) {
      modalInfo.onClose();
    }
    setModalInfo(null);
  };

  const selectedCaptionOption = captionerTypes.find(option => option.name === jobConfig.config.process[0].type);

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
                label="模型名称或路径"
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
                {selectedCaptionOption?.additionalSections?.includes('caption.max_res') && (
                  <div className="mt-4">
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
              </div>
              <div>
                <FormGroup label="选项">
                  <Checkbox
                    label="低显存"
                    checked={jobConfig.config.process[0].caption.low_vram}
                    onChange={value => setJobConfig(value, 'config.process[0].caption.low_vram')}
                  />
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
