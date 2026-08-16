export function sanitizeJobConfigForSave(jobConfig: any) {
  const sanitized = JSON.parse(JSON.stringify(jobConfig));
  const processes = sanitized?.config?.process;
  if (!Array.isArray(processes)) {
    return sanitized;
  }

  for (const processConfig of processes) {
    const trainConfig = processConfig?.train;
    const modelConfig = processConfig?.model;

    if (trainConfig?.cache_text_embeddings && trainConfig?.diff_output_preservation) {
      trainConfig.diff_output_preservation = false;
    }

    const textEncoderPath = modelConfig?.te_name_or_path;
    if (
      modelConfig?.arch === 'ltx2.3:sulphur' &&
      typeof textEncoderPath === 'string' &&
      textEncoderPath.endsWith('gemma_3_12B_it_fp4_mixed.safetensors')
    ) {
      modelConfig.te_name_or_path = null;
    }
  }

  return sanitized;
}
