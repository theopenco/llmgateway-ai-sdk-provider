import type {
  ImageModelV3,
  ImageModelV3CallOptions,
  SharedV3Warning,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';

import { llmgatewayFailedResponseHandler } from '../schemas/error-response';
import type {
  LLMGatewayImageModelId,
  LLMGatewayImageSettings,
} from '../types/llmgateway-image-settings';

type LLMGatewayImageConfig = {
  provider: string;
  headers: () => Record<string, string | undefined>;
  url: (options: { modelId: string; path: string }) => string;
  fetch?: typeof fetch;
};

const LLMGatewayImageResponseSchema = z.object({
  data: z.array(
    z.object({
      b64_json: z.string().optional(),
      url: z.string().optional(),
      revised_prompt: z.string().optional(),
    }),
  ),
});

export class LLMGatewayImageModel implements ImageModelV3 {
  readonly specificationVersion = 'v3' as const;
  readonly provider: string;
  readonly modelId: LLMGatewayImageModelId;
  readonly maxImagesPerCall: number | undefined = undefined;

  readonly settings: LLMGatewayImageSettings;

  private readonly config: LLMGatewayImageConfig;

  constructor(
    modelId: LLMGatewayImageModelId,
    settings: LLMGatewayImageSettings,
    config: LLMGatewayImageConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.provider = config.provider;
  }

  async doGenerate(
    options: ImageModelV3CallOptions,
  ): Promise<{
    images: Array<string>;
    warnings: Array<SharedV3Warning>;
    response: {
      timestamp: Date;
      modelId: string;
      headers: Record<string, string> | undefined;
    };
  }> {
    const warnings: SharedV3Warning[] = [];

    if (options.aspectRatio != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'aspectRatio',
      });
    }

    if (options.seed != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'seed',
      });
    }

    const body: Record<string, unknown> = {
      model: this.modelId,
      prompt: options.prompt,
      n: options.n,
      response_format: 'b64_json',
    };

    if (options.size != null) {
      body.size = options.size;
    }

    const { value: response, responseHeaders } = await postJsonToApi({
      url: this.config.url({
        path: '/images/generations',
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: llmgatewayFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        LLMGatewayImageResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    return {
      images: response.data.map((item) => {
        if (item.b64_json) {
          return item.b64_json;
        }
        throw new Error(
          'Expected b64_json in response but got url. Set response_format to b64_json.',
        );
      }),
      warnings,
      response: {
        timestamp: new Date(),
        modelId: this.modelId,
        headers: responseHeaders as Record<string, string> | undefined,
      },
    };
  }
}
