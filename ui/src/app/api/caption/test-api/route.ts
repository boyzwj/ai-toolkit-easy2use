import { NextResponse } from 'next/server';

type ApiProtocol = 'openai' | 'anthropic';

const TEST_PROMPT = 'Reply with exactly OK.';

function resolveApiEndpoint(baseUrl: string, protocol: ApiProtocol): string {
  const normalized = (baseUrl || '').trim().replace(/\/+$/, '');
  if (!normalized) {
    throw new Error('Base URL 不能为空');
  }

  if (protocol === 'openai') {
    return normalized.endsWith('/chat/completions') ? normalized : `${normalized}/chat/completions`;
  }

  return normalized.endsWith('/messages') ? normalized : `${normalized}/messages`;
}

function buildPayload(modelName: string, protocol: ApiProtocol) {
  if (protocol === 'anthropic') {
    return {
      model: modelName,
      max_tokens: 8,
      messages: [
        {
          role: 'user',
          content: [{ type: 'text', text: TEST_PROMPT }],
        },
      ],
    };
  }

  return {
    model: modelName,
    max_tokens: 8,
    messages: [
      {
        role: 'user',
        content: [{ type: 'text', text: TEST_PROMPT }],
      },
    ],
  };
}

function buildHeaders(apiKey: string, protocol: ApiProtocol): HeadersInit {
  if (protocol === 'anthropic') {
    return {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    };
  }

  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
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

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const modelName = `${body?.model_name_or_path || ''}`.trim();
    const apiBaseUrl = `${body?.api_base_url || ''}`.trim();
    const apiKey = `${body?.api_key || ''}`.trim();
    const apiProtocol = body?.api_protocol === 'anthropic' ? 'anthropic' : 'openai';

    if (!modelName) {
      return NextResponse.json({ error: '模型名称不能为空' }, { status: 400 });
    }
    if (!apiBaseUrl) {
      return NextResponse.json({ error: 'Base URL 不能为空' }, { status: 400 });
    }
    if (!apiKey) {
      return NextResponse.json({ error: 'API Key 不能为空' }, { status: 400 });
    }

    const endpoint = resolveApiEndpoint(apiBaseUrl, apiProtocol);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: buildHeaders(apiKey, apiProtocol),
        body: JSON.stringify(buildPayload(modelName, apiProtocol)),
        signal: controller.signal,
      });
      const rawText = await response.text();

      if (!response.ok) {
        return NextResponse.json(
          {
            success: false,
            error: `测试失败：HTTP ${response.status}`,
            details: rawText.slice(0, 500),
          },
          { status: response.status },
        );
      }

      const preview = parsePreview(rawText);
      return NextResponse.json({
        success: true,
        message: 'API 连通性测试通过',
        endpoint,
        preview,
      });
    } finally {
      clearTimeout(timeout);
    }
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      return NextResponse.json({ success: false, error: '测试超时，请检查网络或接口响应速度' }, { status: 408 });
    }

    return NextResponse.json(
      {
        success: false,
        error: '测试失败，请检查接口配置',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}
