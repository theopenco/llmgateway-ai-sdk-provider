import type { Experimental_VideoModelV4CallOptions } from '@ai-sdk/provider';

import { APICallError } from '@ai-sdk/provider';
import {
  experimental_generateVideo as generateVideo,
  experimental_getVideoStatus as getVideoStatus,
  experimental_startVideo as startVideo,
} from 'ai';
import { vi } from 'vitest';

import { createLLMGateway, llmgateway } from '../index';

const baseURL = 'https://test.llmgateway.io/v1';
const modelId = 'google-vertex/veo-3.1-fast-generate-preview';
const bytes = new Uint8Array([
  0, 0, 0, 24, 102, 116, 121, 112, 109, 112, 52, 50,
]);
const imageBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ';
const imageUrl = 'https://example.com/frame.png';
const callOptions: Experimental_VideoModelV4CallOptions = {
  prompt: 'A waterfall at sunrise',
  n: 1,
  aspectRatio: undefined,
  resolution: '1280x720',
  duration: 8,
  fps: undefined,
  seed: undefined,
  image: undefined,
  frameImages: undefined,
  inputReferences: undefined,
  generateAudio: undefined,
  providerOptions: {},
};

function job(status: string, extra: Record<string, unknown> = {}) {
  return { id: 'v_123', model: modelId, status, error: null, ...extra };
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', 'x-request-id': 'req_123' },
  });
}

function setup(
  responses: Response[] = [
    json(job('queued')),
    json(job('completed')),
    new Response(bytes, { headers: { 'content-type': 'video/mp4' } }),
  ],
) {
  const fetch = vi.fn<typeof globalThis.fetch>(async (_input, init) => {
    init?.signal?.throwIfAborted();
    const response = responses.shift();
    if (!response) {
      throw new Error('Unexpected API call');
    }
    return response;
  });
  const provider = createLLMGateway({
    apiKey: 'test-key',
    baseURL: `${baseURL}/`,
    headers: { 'x-provider': 'provider' },
    fetch,
  });
  const model = provider.video(modelId);
  const generate = (
    options: Partial<Parameters<typeof generateVideo>[0]> = {},
  ) =>
    generateVideo({
      model,
      prompt: 'A waterfall at sunrise',
      duration: 8,
      maxRetries: 0,
      poll: { intervalMs: 0 },
      ...options,
    });
  return { provider, model, fetch, generate };
}

describe('LLMGateway video', () => {
  it('exposes v4 video factories, aliases, settings, and arbitrary model IDs', () => {
    const { provider } = setup();
    expect(provider.videoModel).toBe(provider.video);
    expect(llmgateway.video('future/video-model')).toMatchObject({
      specificationVersion: 'v4',
      provider: 'llmgateway.video',
      modelId: 'future/video-model',
      maxVideosPerCall: 1,
    });
    expect(
      provider.video('kling-v3-0', { extraBody: { audio: false } }).settings,
    ).toEqual({ extraBody: { audio: false } });
  });

  it('generates video through the SDK, polls pending states, and downloads authenticated bytes', async () => {
    const { generate, fetch } = setup([
      json(job('queued')),
      json(job('queued')),
      json(job('in_progress')),
      json(
        job('completed', {
          content: [
            { type: 'video', url: 'https://upstream.example/private.mp4' },
          ],
        }),
      ),
      new Response(bytes, {
        headers: { 'content-type': 'video/webm; charset=binary' },
      }),
    ]);
    const abortSignal = new AbortController().signal;
    const result = await generate({
      resolution: '1920x1080',
      generateAudio: false,
      headers: { 'x-request': 'request' },
      abortSignal,
    });
    expect(result.video.uint8Array).toEqual(bytes);
    expect(result.video.mediaType).toBe('video/webm');
    expect(result.video.base64).toBe('AAAAGGZ0eXBtcDQy');
    expect(result.warnings).toEqual([]);
    expect(result.providerMetadata?.llmgateway?.videos).toEqual([
      { id: 'v_123', status: 'completed' },
    ]);
    expect(result.responses[0]).toMatchObject({
      timestamp: expect.any(Date),
      modelId,
      headers: { 'x-request-id': 'req_123' },
    });
    expect(fetch.mock.calls.map(([url, init]) => [url, init?.method])).toEqual([
      [`${baseURL}/videos`, 'POST'],
      [`${baseURL}/videos/v_123`, 'GET'],
      [`${baseURL}/videos/v_123`, 'GET'],
      [`${baseURL}/videos/v_123`, 'GET'],
      [`${baseURL}/videos/v_123/content`, 'GET'],
    ]);
    expect(JSON.parse(fetch.mock.calls[0]![1]!.body as string)).toEqual({
      model: modelId,
      prompt: 'A waterfall at sunrise',
      seconds: 8,
      size: '1920x1080',
      audio: false,
    });
    for (const [, init] of fetch.mock.calls) {
      const headers = new Headers(init?.headers);
      expect(headers.get('authorization')).toBe('Bearer test-key');
      expect(headers.get('x-provider')).toBe('provider');
      expect(headers.get('x-request')).toBe('request');
      expect(init?.signal).toBe(abortSignal);
    }
    expect(
      new Headers(fetch.mock.calls[0]![1]?.headers).get('idempotency-key'),
    ).toMatch(/^aisdk_vid_/);
  });

  it('supports start/status calls with a JSON-serializable job ID', async () => {
    const { model } = setup();
    const started = await startVideo({
      model,
      prompt: 'A waterfall',
      duration: 8,
    });
    expect(started.operation).toBe('v_123');
    const result = await getVideoStatus(model, {
      operation: JSON.parse(JSON.stringify(started.operation)),
    });
    expect(result.status).toBe('completed');
    if (result.status === 'completed') {
      expect(result.videos[0]).toEqual({
        type: 'binary',
        data: bytes,
        mediaType: 'video/mp4',
      });
    }
  });

  it('generates n videos as separate jobs', async () => {
    const fetch = vi.fn<typeof globalThis.fetch>(async (url, init) => {
      if (init?.method === 'POST') {
        return json(job('queued'));
      }
      if (String(url).endsWith('/content')) {
        return new Response(bytes, {
          headers: { 'content-type': 'video/mp4' },
        });
      }
      return json(job('completed'));
    });
    const provider = createLLMGateway({ apiKey: 'test', fetch });
    const result = await generateVideo({
      model: provider.video(modelId),
      prompt: 'Ocean waves',
      duration: 8,
      n: 2,
      poll: { intervalMs: 0 },
    });
    expect(result.videos).toHaveLength(2);
    expect(
      fetch.mock.calls.filter(([, init]) => init?.method === 'POST'),
    ).toHaveLength(2);
  });

  it.each([
    ['URL', { type: 'url' as const, url: imageUrl }, imageUrl],
    [
      'base64',
      { type: 'file' as const, mediaType: 'image/png', data: imageBase64 },
      `data:image/png;base64,${imageBase64}`,
    ],
    [
      'bytes',
      {
        type: 'file' as const,
        mediaType: 'image/png',
        data: new Uint8Array([0, 1, 2]),
      },
      'data:image/png;base64,AAEC',
    ],
  ])('maps %s image input', async (_name, image, expected) => {
    const { model, fetch } = setup();
    await model.doStart({ ...callOptions, image });
    expect(JSON.parse(fetch.mock.calls[0]![1]!.body as string).image).toEqual({
      image_url: expected,
    });
  });

  it('normalizes image-to-video prompts through generateVideo', async () => {
    const { generate, fetch } = setup();
    await generate({ prompt: { image: imageUrl, text: 'Animate this frame' } });
    expect(JSON.parse(fetch.mock.calls[0]![1]!.body as string)).toMatchObject({
      prompt: 'Animate this frame',
      image: { image_url: imageUrl },
    });
  });

  it('maps first and last frame images', async () => {
    const { generate, fetch } = setup();
    await generate({
      frameImages: [
        { frameType: 'first_frame', image: imageUrl },
        { frameType: 'last_frame', image: 'https://example.com/last.png' },
      ],
    });
    expect(JSON.parse(fetch.mock.calls[0]![1]!.body as string)).toMatchObject({
      image: { image_url: imageUrl },
      last_frame: { image_url: 'https://example.com/last.png' },
    });
  });

  it('maps image and video references by media type', async () => {
    const { generate, fetch } = setup();
    await generate({
      inputReferences: [
        { data: imageUrl, mediaType: 'image/png' },
        { data: 'https://example.com/motion.mp4', mediaType: 'video/mp4' },
      ],
    });
    expect(JSON.parse(fetch.mock.calls[0]![1]!.body as string)).toMatchObject({
      reference_images: [{ image_url: imageUrl }],
      reference_videos: [{ video_url: 'https://example.com/motion.mp4' }],
    });
  });

  it('merges provider, model, and call options in precedence order', async () => {
    const { fetch } = setup();
    const provider = createLLMGateway({
      apiKey: 'test',
      baseURL,
      fetch,
      extraBody: {
        audio: true,
        seconds: 4,
        callback_url: 'https://example.com/webhook',
      },
    });
    const model = provider.video(modelId, {
      extraBody: { seconds: 6, callback_secret: 'test-secret' },
    });
    await model.doStart({
      ...callOptions,
      providerOptions: { llmgateway: { seconds: 10, audio: false } },
    });
    expect(JSON.parse(fetch.mock.calls[0]![1]!.body as string)).toMatchObject({
      seconds: 10,
      audio: false,
      callback_url: 'https://example.com/webhook',
      callback_secret: 'test-secret',
    });
  });

  it('warns about unsupported options without sending them', async () => {
    const { model, fetch } = setup();
    const result = await model.doStart({
      ...callOptions,
      aspectRatio: '9:16',
      fps: 24,
      seed: 0,
    });
    expect(
      result.warnings.map((w) => (w.type === 'unsupported' ? w.feature : '')),
    ).toEqual(['aspectRatio', 'fps', 'seed']);
    const body = JSON.parse(fetch.mock.calls[0]![1]!.body as string);
    expect(body).not.toHaveProperty('aspect_ratio');
    expect(body).not.toHaveProperty('fps');
    expect(body).not.toHaveProperty('seed');
  });

  it.each([undefined, 0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects invalid duration %s before submission',
    async (duration) => {
      const { model, fetch } = setup();
      await expect(model.doStart({ ...callOptions, duration })).rejects.toThrow(
        'positive integer duration',
      );
      expect(fetch).not.toHaveBeenCalled();
    },
  );

  it.each([undefined, ''])('requires a text prompt', async (prompt) => {
    const { model, fetch } = setup();
    await expect(model.doStart({ ...callOptions, prompt })).rejects.toThrow(
      'requires a text prompt',
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects overriding the one-video API limit', async () => {
    const { generate, fetch } = setup();
    await expect(generate({ n: 2, maxVideosPerCall: 2 })).rejects.toThrow(
      'multiple videos per API call',
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects duplicate frame inputs', async () => {
    const { model, fetch } = setup();
    await expect(
      model.doStart({
        ...callOptions,
        frameImages: [
          { frameType: 'first_frame', image: { type: 'url', url: imageUrl } },
          { frameType: 'first_frame', image: { type: 'url', url: imageUrl } },
        ],
      }),
    ).rejects.toThrow('Only one first_frame');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects inline reference videos', async () => {
    const { model, fetch } = setup();
    await expect(
      model.doStart({
        ...callOptions,
        inputReferences: [
          { type: 'file', mediaType: 'video/mp4', data: bytes },
        ],
      }),
    ).rejects.toThrow('video references other than HTTPS URLs');
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(['failed', 'canceled', 'expired'])(
    'surfaces %s jobs without downloading',
    async (status) => {
      const { generate, fetch } = setup([
        json(job('queued')),
        json(job(status)),
      ]);
      await expect(generate()).rejects.toThrow(`Video generation ${status}`);
      expect(fetch).toHaveBeenCalledTimes(2);
    },
  );

  it('surfaces the upstream job error message', async () => {
    const { generate } = setup([
      json(job('queued')),
      json(
        job('failed', {
          error: {
            code: 'content_filter',
            message: 'Generation was blocked by the model',
          },
        }),
      ),
    ]);
    await expect(generate()).rejects.toThrow(
      'Generation was blocked by the model',
    );
  });

  it('does not retry a job that fails at creation', async () => {
    const { generate, fetch } = setup([
      json(job('failed', { error: { message: 'Quota exhausted' } })),
    ]);
    await expect(generate({ maxRetries: 2 })).rejects.toMatchObject({
      isRetryable: false,
      message: 'Quota exhausted',
    });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it.each(['create', 'status', 'content'])(
    'propagates HTTP failures from %s',
    async (stage) => {
      const error = json(
        { error: { message: 'Access denied', code: 'unauthorized' } },
        401,
      );
      const responses =
        stage === 'create'
          ? [error]
          : stage === 'status'
            ? [json(job('queued')), error]
            : [json(job('queued')), json(job('completed')), error];
      const { generate } = setup(responses);
      await expect(generate()).rejects.toBeInstanceOf(APICallError);
    },
  );

  it.each(['create', 'status'])(
    'rejects malformed %s responses',
    async (stage) => {
      const malformed = json({ status: 'unknown' });
      const { generate } = setup(
        stage === 'create' ? [malformed] : [json(job('queued')), malformed],
      );
      await expect(generate()).rejects.toThrow('Invalid JSON response');
    },
  );

  it.each([
    new Response(new Uint8Array(), {
      headers: { 'content-type': 'video/mp4' },
    }),
    new Response('not a video', {
      headers: { 'content-type': 'application/json' },
    }),
  ])('rejects empty or non-video content', async (content) => {
    const { generate } = setup([
      json(job('queued')),
      json(job('completed')),
      content,
    ]);
    await expect(generate()).rejects.toThrow(
      'Expected non-empty video content',
    );
  });

  it('uses mp4 when the content endpoint omits its media type', async () => {
    const { generate } = setup([
      json(job('queued')),
      json(job('completed')),
      new Response(bytes),
    ]);
    expect((await generate()).video.mediaType).toBe('video/mp4');
  });

  it('accepts generic binary content from upstream video downloads', async () => {
    const { generate } = setup([
      json(job('queued')),
      json(job('completed')),
      new Response(bytes, {
        headers: { 'content-type': 'application/octet-stream' },
      }),
    ]);
    const result = await generate();
    expect(result.video.mediaType).toBe('video/mp4');
    expect(result.video.uint8Array).toEqual(bytes);
  });

  it('warns when a webhook URL is supplied without signed callback settings', async () => {
    const { model } = setup();
    const result = await model.doStart({
      ...callOptions,
      webhookUrl: 'https://example.com/webhook',
    });
    expect(result.warnings).toContainEqual(
      expect.objectContaining({ type: 'unsupported', feature: 'webhookUrl' }),
    );
  });

  it('encodes the job ID before constructing authenticated URLs', async () => {
    const { model, fetch } = setup([json(job('in_progress'))]);
    await model.doStatus({ operation: 'v/123?other=true' });
    expect(fetch.mock.calls[0]?.[0]).toBe(
      `${baseURL}/videos/v%2F123%3Fother%3Dtrue`,
    );
  });

  it('rejects invalid operation handles', async () => {
    const { model, fetch } = setup();
    await expect(
      model.doStatus({ operation: { url: 'https://example.com' } }),
    ).rejects.toThrow('Expected a video job ID');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('honors SDK polling timeouts', async () => {
    const { generate, fetch } = setup([json(job('queued'))]);
    await expect(generate({ poll: { timeoutMs: 0 } })).rejects.toThrow(
      'Video generation timed out',
    );
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('aborts polling without making another request', async () => {
    const controller = new AbortController();
    const { generate, fetch } = setup([json(job('queued'))]);
    await expect(
      generate({
        abortSignal: controller.signal,
        poll: {
          delay: async () => {
            controller.abort();
            controller.signal.throwIfAborted();
          },
        },
      }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('forwards an aborted signal to the content download', async () => {
    const controller = new AbortController();
    const { model, fetch } = setup();
    fetch.mockImplementationOnce(async () => {
      controller.abort();
      return json(job('completed'));
    });
    await expect(
      model.doStatus({ operation: 'v_123', abortSignal: controller.signal }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetch.mock.calls[1]?.[1]?.signal).toBe(controller.signal);
  });
});
