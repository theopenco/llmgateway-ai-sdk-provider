import { createLLMGateway } from '../provider';
import { createTestServer } from '../tests/create-test-server';

const TEST_BASE64_IMAGE =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

const testUrl = 'https://test.llmgateway.io/v1/images/generations';

const server = createTestServer({
  [testUrl]: {
    response: {
      type: 'json-value',
      body: {
        data: [{ b64_json: TEST_BASE64_IMAGE }],
      },
    },
  },
});

const provider = createLLMGateway({
  apiKey: 'test-api-key',
  baseURL: 'https://test.llmgateway.io/v1',
  compatibility: 'strict',
  fetch: server.fetch,
});

const model = provider.image('qwen-image-plus');

describe('LLMGatewayImageModel', () => {
  beforeEach(() => {
    server.calls.length = 0;
  });

  describe('doGenerate', () => {
    it('should send correct request body', async () => {
      await model.doGenerate({
        prompt: 'a cute cat',
        n: 1,
        size: '1024x1024',
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      const body = await server.calls[0]!.requestBodyJson;
      expect(body).toEqual({
        model: 'qwen-image-plus',
        prompt: 'a cute cat',
        n: 1,
        size: '1024x1024',
        response_format: 'b64_json',
      });
    });

    it('should return base64 image data', async () => {
      const result = await model.doGenerate({
        prompt: 'a cute cat',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.images).toEqual([TEST_BASE64_IMAGE]);
    });

    it('should not include size in body when undefined', async () => {
      await model.doGenerate({
        prompt: 'a sunset',
        n: 2,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      const body = await server.calls[0]!.requestBodyJson;
      expect(body).toEqual({
        model: 'qwen-image-plus',
        prompt: 'a sunset',
        n: 2,
        response_format: 'b64_json',
      });
    });

    it('should warn for unsupported aspectRatio', async () => {
      const result = await model.doGenerate({
        prompt: 'a dog',
        n: 1,
        size: undefined,
        aspectRatio: '16:9',
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.warnings).toContainEqual({
        type: 'unsupported',
        feature: 'aspectRatio',
      });
    });

    it('should warn for unsupported seed', async () => {
      const result = await model.doGenerate({
        prompt: 'a dog',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: 42,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.warnings).toContainEqual({
        type: 'unsupported',
        feature: 'seed',
      });
    });

    it('should send authorization header', async () => {
      await model.doGenerate({
        prompt: 'a tree',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(server.calls[0]!.requestHeaders.authorization).toBe(
        'Bearer test-api-key',
      );
    });

    it('should return multiple images', async () => {
      const multiServer = createTestServer({
        [testUrl]: {
          response: {
            type: 'json-value',
            body: {
              data: [
                { b64_json: TEST_BASE64_IMAGE },
                { b64_json: TEST_BASE64_IMAGE },
              ],
            },
          },
        },
      });

      const multiModel = createLLMGateway({
        apiKey: 'test-api-key',
        baseURL: 'https://test.llmgateway.io/v1',
        compatibility: 'strict',
        fetch: multiServer.fetch,
      }).image('qwen-image-plus');

      const result = await multiModel.doGenerate({
        prompt: 'two cats',
        n: 2,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.images).toHaveLength(2);
    });

    it('should include response metadata', async () => {
      const result = await model.doGenerate({
        prompt: 'a bird',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.response.modelId).toBe('qwen-image-plus');
      expect(result.response.timestamp).toBeInstanceOf(Date);
    });

    it('should set specificationVersion to v3', () => {
      expect(model.specificationVersion).toBe('v3');
    });

    it('should set provider to llmgateway.image', () => {
      expect(model.provider).toBe('llmgateway.image');
    });
  });
});
