import { SelectOption } from '@/types';

export const defaultCaptionPromptTemplate = '详细描述 (Detailed Description)';
export const defaultCaptionTargetLanguage = '英语 (English)';

export const captionPromptTemplates: Record<string, string> = {
  '标签生成 (Tag Generation)':
    "Your task is to generate a clean list of comma-separated tags for a text-to-image AI, based *only* on the visual information in the image. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
  '简单描述 (Short Description)':
    'Analyze the image and write a single concise sentence that describes the main subject and setting. Keep it grounded in visible details only.',
  '详细描述 (Detailed Description)':
    'Generate a detailed paragraph that combines the subject, actions, environment, lighting, and mood into 2-3 cohesive sentences. Focus on accurate visual details rather than speculation.',
  '超详尽描述 (Extremely Detailed)':
    'Produce an extremely rich description touching on appearance, clothing textures, background elements, light quality, shadows, and atmosphere. Aim for an immersive depiction rooted in what the image shows.',
  '电影感描述 (Cinematic)':
    'Describe the scene as if capturing a cinematic shot. Cover subject, pose, environment, lighting, mood, and artistic style (photorealistic, painterly, etc.) in one vivid paragraph emphasizing visual impact.',
  '详细分析 (Analysis)':
    'Describe this image in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.',
  // 视频打标暂未开放，先保留模板定义以便后续恢复。
  // '视频摘要 (Video Summary)': 'Summarize the key events and narrative points in this video.',
  // '短篇故事 (Story)': 'Write a short, imaginative story inspired by this image or video.',
  'Danbooru Tags (Anime)': 'Describe this image using Danbooru tags, separated by commas.',
};

export const captionPromptTemplateOptions: SelectOption[] = Object.keys(captionPromptTemplates).map(key => ({
  value: key,
  label: key,
}));

export const captionTargetLanguageOptions: SelectOption[] = [
  { value: '英语 (English)', label: '英语 (English)' },
  { value: '中文 (Chinese)', label: '中文 (Chinese)' },
  { value: '中英双语 (Bilingual)', label: '中英双语 (Bilingual)' },
];

export function buildCaptionPrompt(templateKey?: string, targetLanguage?: string) {
  const safeTemplateKey =
    templateKey && captionPromptTemplates[templateKey] ? templateKey : defaultCaptionPromptTemplate;
  const safeTargetLanguage = targetLanguage || defaultCaptionTargetLanguage;

  const baseText = captionPromptTemplates[safeTemplateKey] || captionPromptTemplates[defaultCaptionPromptTemplate];
  let suffix = '';

  if (safeTargetLanguage.includes('中文') && !safeTargetLanguage.includes('双语')) {
    suffix = '\n\n请直接输出中文描述，不要包含任何开场白（如“好的”、“这是一段描述”等）或结束语。';
  } else if (safeTargetLanguage.includes('双语')) {
    suffix =
      '\n\n请提供中文和英文双语描述。\n' +
      '要求格式严格如下：\n' +
      '## Chinese Description\n' +
      '[中文内容]\n\n' +
      '## English Description\n' +
      '[English Content]\n\n' +
      '注意：不要包含任何开场白或多余的解释性文字，直接输出内容。';
  } else {
    suffix = "\n\nOutput the description directly without any conversational fillers (e.g., 'Here is a description').";
  }

  return `${baseText}${suffix}`;
}
