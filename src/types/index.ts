import type { LanguageModelV1 } from '@ai-sdk/provider';

// Re-export the LanguageModelV1 type to ensure proper type compatibility
export type { LanguageModelV1 };

// Export our model types with explicit type constraints
export type LLMGatewayLanguageModel = LanguageModelV1;

export type LLMGatewayProviderOptions = {
  models?: string[];

  /**
   * https://llmgateway.io/docs/use-cases/reasoning-tokens
   * One of `max_tokens` or `effort` is required.
   * If `exclude` is true, reasoning will be removed from the response. Default is false.
   */
  reasoning?: {
    exclude?: boolean;
  } & (
    | {
        max_tokens: number;
      }
    | {
        effort: 'high' | 'medium' | 'low';
      }
  );

  /**
   * A unique identifier representing your end-user, which can
   * help LLMGateway to monitor and detect abuse.
   */
  user?: string;
};

export type LLMGatewaySharedSettings = LLMGatewayProviderOptions & {
  /**
   * @deprecated use `reasoning` instead
   */
  includeReasoning?: boolean;

  extraBody?: Record<string, unknown>;

  /**
   * Enable usage accounting to get detailed token usage information.
   * https://llmgateway.io/docs/use-cases/usage-accounting
   */
  usage?: {
    /**
     * When true, includes token usage information in the response.
     */
    include: boolean;
  };
};

/**
 * Usage accounting response
 * @see https://llmgateway.io/docs/use-cases/usage-accounting
 */
export type LLMGatewayUsageAccounting = {
  promptTokens: number;
  promptTokensDetails?: {
    cachedTokens: number;
  };
  completionTokens: number;
  completionTokensDetails?: {
    reasoningTokens: number;
  };
  totalTokens: number;
  cost?: number;
};
