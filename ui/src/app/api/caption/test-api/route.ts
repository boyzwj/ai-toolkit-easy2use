import { NextResponse } from 'next/server';

type ApiProtocol = 'openai' | 'anthropic';

const TEST_PROMPT = 'Reply with exactly OK.';

function buildOpenaiHeaders(apiKey: string): HeadersInit {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
  };
}

function buildAnthropicHeaders(apiKey: string): HeadersInit {
  return {
    'Content-Type': 'application/json',
    'x-api-key': apiKey,
    'anthropic-version': '2023-06-01',
  };
}

function parsePreview(rawText: string) {
  try {
    const payload = JSON.parse(rawText);

    const openaiText = payload?.choices?.[0]?.message?.content;
    if (typeof openaiText === 'string' && openaiText.trim()) {
      return openaiText.trim();
    }

    const anthropicContent = payload?.content;
    if (Array.isArray(anthropicContent)) {
      const text = anthropicContent
        .filter((item: any) => item?.type === 'text' && typeof item?.text === 'string')
        .map((item: any) => item.text.trim())
        .filter(Boolean)
        .join('\n');
      if (text) {
        return text;
      }
    }

    return JSON.stringify(payload).slice(0, 200);
  } catch {
    return rawText.trim().slice(0, 200);
  }
}

async function fetchJson(url: string, headers: HeadersInit, timeoutMs = 10000): Promise<any> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { headers, signal: controller.signal });
    const text = await res.text();
    try {
      return res.ok ? JSON.parse(text) : null;
    } catch {
      return null;
    }
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function tryFetchModels(apiBaseUrl: string, apiKey: string): Promise<string[]> {
  const base = apiBaseUrl.replace(/\/+$/, '');
  // Standard OpenAI: GET /v1/models
  const modelsUrl = `${base}/models`;
  const data = await fetchJson(modelsUrl, buildOpenaiHeaders(apiKey), 10000);
  if (!data) return [];

  // OpenAI format: { data: [{ id: "..." }] }
  if (data.data && Array.isArray(data.data)) {
    const ids = data.data.filter((m: any) => m?.id).map((m: any) => m.id);
    if (ids.length > 0) return ids;
  }
  // Ollama format: { models: [{ name: "..." }] }
  if (data.models && Array.isArray(data.models)) {
    const names = data.models.filter((m: any) => m?.name).map((m: any) => m.name);
    if (names.length > 0) return names;
  }

  return [];
}

async function tryHealthCheck(apiBaseUrl: string, apiKey: string): Promise<boolean> {
  const base = apiBaseUrl.replace(/\/+$/, '');
  // Try /health first (common for vLLM, LocalAI etc.)
  // Go up one level from /v1 if present, since /health is usually at root
  const root = base.replace(/\/v1\/?$/, '');
  const candidates = [
    `${root}/health`,
    `${base}/health`,
    root || base,
  ];
  for (const url of [...new Set(candidates)]) {
    const data = await fetchJson(url, buildOpenaiHeaders(apiKey), 5000);
    if (data !== null) return true;
  }
  return false;
}

async function tryChatCompletion(apiBaseUrl: string, apiKey: string, modelName: string): Promise<string | null> {
  const base = apiBaseUrl.replace(/\/+$/, '');
  const endpoint = base.endsWith('/chat/completions') ? base : `${base}/chat/completions`;
  const body = {
    model: modelName || 'test',
    max_tokens: 8,
    messages: [{ role: 'user', content: [{ type: 'text', text: TEST_PROMPT }] }],
  };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: buildOpenaiHeaders(apiKey),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) return null;
    const text = await res.text();
    return parsePreview(text);
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const modelName = `${body?.model_name_or_path || ''}`.trim();
    const apiBaseUrl = `${body?.api_base_url || ''}`.trim();
    const apiKey = `${body?.api_key || ''}`.trim();
    const apiProtocol: ApiProtocol = body?.api_protocol === 'anthropic' ? 'anthropic' : 'openai';

    if (!apiBaseUrl) {
      return NextResponse.json({ error: 'Base URL 不能为空' }, { status: 400 });
    }
    if (!apiKey) {
      return NextResponse.json({ error: 'API Key 不能为空' }, { status: 400 });
    }

    if (apiProtocol === 'anthropic') {
      // Anthropic: use /messages for connectivity test
      const endpoint = apiBaseUrl.replace(/\/+$/, '');
      const messagesUrl = endpoint.endsWith('/messages') ? endpoint : `${endpoint}/messages`;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 20000);
      try {
        const res = await fetch(messagesUrl, {
          method: 'POST',
          headers: buildAnthropicHeaders(apiKey),
          body: JSON.stringify({
            model: modelName || 'claude-3-haiku-20240307',
            max_tokens: 8,
            messages: [{ role: 'user', content: [{ type: 'text', text: TEST_PROMPT }] }],
          }),
          signal: controller.signal,
        });
        const rawText = await res.text();
        if (!res.ok) {
          return NextResponse.json(
            { success: false, error: `测试失败：HTTP ${res.status}`, details: rawText.slice(0, 500) },
            { status: res.status },
          );
        }
        const preview = parsePreview(rawText);
        return NextResponse.json({ success: true, message: 'API 连通性测试通过', preview, models: [] });
      } finally {
        clearTimeout(timeout);
      }
    }

    // OpenAI protocol: try /health, /models (connectivity + model listing)
    const [healthOk, models] = await Promise.all([
      tryHealthCheck(apiBaseUrl, apiKey),
      tryFetchModels(apiBaseUrl, apiKey),
    ]);

    // If /health failed, try /chat/completions as fallback
    let preview: string | null = null;
    if (!healthOk) {
      preview = await tryChatCompletion(apiBaseUrl, apiKey, modelName);
    }

    if (!healthOk && !preview) {
      return NextResponse.json(
        {
          success: false,
          error: '测试失败：无法连接到 API 服务',
          details: '请检查 Base URL 和 API Key 是否正确，以及服务是否正常运行。',
        },
        { status: 400 },
      );
    }

    return NextResponse.json({
      success: true,
      message: 'API 连通性测试通过',
      preview: preview || '',
      models,
    });
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      return NextResponse.json({ success: false, error: '测试超时，请检查网络或接口响应速度' }, { status: 408 });
    }
    return NextResponse.json(
      { success: false, error: '测试失败，请检查接口配置', details: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
}
