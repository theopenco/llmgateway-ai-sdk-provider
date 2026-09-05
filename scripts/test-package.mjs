import assert from 'node:assert/strict';
import { createLLMGateway } from '@llmgateway/ai-sdk-provider';
import {
  LLMGatewayChatLanguageModel,
  LLMGatewayCompletionLanguageModel,
  LLMGatewayImageModel,
  LLMGatewayVideoModel,
} from '@llmgateway/ai-sdk-provider/internal';
import { experimental_generateVideo as generateVideo } from 'ai';

const videoBytes = new Uint8Array([0, 0, 0, 24, 102, 116, 121, 112]);
const provider = createLLMGateway({
  apiKey: 'package-test',
  fetch: async (url, init) => {
    assert.equal(
      new Headers(init?.headers).get('authorization'),
      'Bearer package-test',
    );
    if (String(url).endsWith('/content')) {
      return new Response(videoBytes, {
        headers: { 'content-type': 'video/mp4' },
      });
    }
    return Response.json({
      id: 'v_package_test',
      model: 'veo-3.1-fast-generate-preview',
      status: init?.method === 'POST' ? 'queued' : 'completed',
      error: null,
    });
  },
});

// Both package entry points must share the same class definitions.
assert.ok(provider.chat('test') instanceof LLMGatewayChatLanguageModel);
assert.ok(
  provider.completion('test') instanceof LLMGatewayCompletionLanguageModel,
);
assert.ok(provider.image('test') instanceof LLMGatewayImageModel);
assert.ok(provider.video('test') instanceof LLMGatewayVideoModel);

const result = await generateVideo({
  model: provider.video('veo-3.1-fast-generate-preview'),
  prompt: 'Ocean waves',
  duration: 8,
  poll: { intervalMs: 0 },
});
assert.deepEqual(result.video.uint8Array, videoBytes);
assert.equal(result.video.mediaType, 'video/mp4');
