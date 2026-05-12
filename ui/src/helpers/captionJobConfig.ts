import { CaptionJobConfig } from "@/types";
import { captionerTypes } from "./captionOptions";
import { defaultCaptionPromptTemplate, defaultCaptionTargetLanguage, buildCaptionPrompt } from './captionPrompts';


const defaultImageCaptionPrompt = buildCaptionPrompt(defaultCaptionPromptTemplate, defaultCaptionTargetLanguage);

export const defaultCaptionJobConfig: CaptionJobConfig = {
  job: 'extension',
  config: {
    name: 'Caption Directory',
    process: [
      {
        type: 'RemoteAPICaptioner',
        device: 'cuda',
        caption: {
          model_name_or_path: '',
          api_base_url: '',
          api_key: '',
          api_protocol: 'openai',
          prompt_template: defaultCaptionPromptTemplate,
          target_lang: defaultCaptionTargetLanguage,
          caption_prompt: defaultImageCaptionPrompt,
          extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'],
          path_to_caption: '',
          recaption: false,
          max_res: 512,
          max_new_tokens: 128,
          api_concurrency: 20,
          dtype: 'bf16',
          qtype: 'float8',
          quantize: false,
          low_vram: false,
        },
      },
    ],
  },
};


const repairDefaults = (defaults: { [key: string]: any }) => {
  let newDefaults: { [key: string]: any } = {};
  // if the key doesnt start with config.process[0]., then add it
  for (const key in defaults) {
    if (!key.startsWith('config.process[0].')) {
      newDefaults[`config.process[0].${key}`] = defaults[key];
    } else {
      newDefaults[key] = defaults[key];
    }
  }
  return newDefaults;
}



export const handleCaptionerTypeChange = (
  currentTypeName: string,
  newTypeName: string,
  jobConfig: CaptionJobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const currentType = captionerTypes.find(a => a.name === currentTypeName);
  if (!currentType || currentType.name === newTypeName) {
    return;
  }

  // update the defaults when a model is selected
  const newType = captionerTypes.find(model => model.name === newTypeName);

  let currentDefaults = repairDefaults(currentType.defaults || {});
  let newDefaults = repairDefaults(newType?.defaults || {});

  // set new model
  setJobConfig(newTypeName, 'config.process[0].type');

  // revert defaults from previous model
  for (const key in currentDefaults) {
    setJobConfig(currentDefaults[key][1], key);
  }

  for (const key in newDefaults) {
    setJobConfig(newDefaults[key][0], key);
  }
};
