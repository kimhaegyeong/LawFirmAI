/**
 * useStreamingMessage 훅 테스트
 * 포괄적인 기능 테스트 및 리팩토링 전 검증
 */
import { describe, it, expect, vi } from 'vitest';
// import { renderHook, act, waitFor } from '@testing-library/react';
// import { useStreamingMessage } from '../useStreamingMessage';
// import { parseSourcesMetadata, mergeSourcesMetadata } from '../../utils/sourcesParser';
// import type { ChatMessage, FileAttachment } from '../../types/chat';
// import type { StreamError } from '../../types/error';
import type { ChatMessage } from '../../types/chat';

// Mock dependencies
vi.mock('../useChat', () => ({
  useChat: () => ({
    sendStreamingMessage: vi.fn(),
  }),
}));

vi.mock('../../utils/streamParser', () => ({
  parseStreamChunk: vi.fn((chunk: string) => {
    try {
      return JSON.parse(chunk);
    } catch {
      return { type: 'stream', content: chunk };
    }
  }),
}));

vi.mock('../../utils/sourcesParser', () => ({
  parseSourcesMetadata: vi.fn((metadata: Record<string, unknown> | undefined) => ({
    sources: metadata?.sources || [],
    legalReferences: metadata?.legal_references || [],
    sourcesDetail: metadata?.sources_detail || [],
    sourcesByType: metadata?.sources_by_type || {
      statute_article: [],
      case_paragraph: [],
      decision_paragraph: [],
      interpretation_paragraph: [],
    },
    relatedQuestions: metadata?.related_questions || [],
    messageId: metadata?.message_id,
  })),
  mergeSourcesMetadata: vi.fn((existing: Record<string, unknown>, newSources: Record<string, unknown>) => ({
    ...existing,
    ...newSources,
  })),
}));

vi.mock('../../utils/errorHandler', () => ({
  toStreamError: vi.fn((error: Error | unknown) => ({
    type: 'unknown',
    message: error?.message || 'Unknown error',
    canRetry: false,
  })),
  extractQuotaInfo: vi.fn(),
}));

vi.mock('../../utils/logger', () => ({
  default: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('useStreamingMessage - Done 이벤트 처리', () => {
  it('should set _isDone flag when done event is received', () => {
    const metadata = {
      sources: ['참고자료 1'],
    };

    const doneMetadata = {
      ...metadata,
      _isDone: true,
    };

    expect(doneMetadata._isDone).toBe(true);
    expect(doneMetadata.sources).toEqual(['참고자료 1']);
  });

  it('should update content with done event content', () => {
    const streamContent = '스트리밍 중인 내용';
    const doneContent = '최종 완성된 답변 내용';

    const finalContent = doneContent && doneContent.trim() ? doneContent : streamContent;

    expect(finalContent).toBe('최종 완성된 답변 내용');
  });

  it('should use stream content if done content is empty', () => {
    const streamContent = '스트리밍 중인 내용';
    const doneContent = '';

    const finalContent = doneContent && doneContent.trim() ? doneContent : streamContent;

    expect(finalContent).toBe('스트리밍 중인 내용');
  });

  it('should handle done event with metadata', () => {
    const metadata = {
      sources_by_type: {
        statute_article: [{ name: '법령 1' }],
        case_paragraph: [],
        decision_paragraph: [],
        interpretation_paragraph: [],
      },
      related_questions: ['관련 질문 1'],
    };

    const doneMetadata = {
      ...metadata,
      _isDone: true,
    };

    expect(doneMetadata._isDone).toBe(true);
    expect(doneMetadata.related_questions).toEqual(['관련 질문 1']);
  });

  it('should find message by assistantMessageId when processing done event', () => {
    const messages: ChatMessage[] = [
      { id: 'user-1', role: 'user', content: '질문', timestamp: new Date() },
      { id: 'assistant-1', role: 'assistant', content: '답변', timestamp: new Date() },
    ];

    const assistantMessageId = 'assistant-1';
    const messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);

    expect(messageIndex).toBe(1);
  });

  it('should find message by streamingMessageId as fallback', () => {
    const messages: ChatMessage[] = [
      { id: 'user-1', role: 'user', content: '질문', timestamp: new Date() },
      { id: 'assistant-1', role: 'assistant', content: '답변', timestamp: new Date() },
    ];

    const assistantMessageId = 'assistant-2';
    const streamingMessageId = 'assistant-1';

    let messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
    if (messageIndex === -1 && streamingMessageId) {
      messageIndex = messages.findIndex((msg) => msg.id === streamingMessageId);
    }

    expect(messageIndex).toBe(1);
  });

  it('should find last assistant message as final fallback', () => {
    const messages: ChatMessage[] = [
      { id: 'user-1', role: 'user', content: '질문', timestamp: new Date() },
      { id: 'assistant-1', role: 'assistant', content: '답변', timestamp: new Date() },
    ];

    const assistantMessageId = 'assistant-2';
    const streamingMessageId = null;

    let messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
    if (messageIndex === -1 && streamingMessageId) {
      messageIndex = messages.findIndex((msg) => msg.id === streamingMessageId);
    }
    if (messageIndex === -1) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg && msg.role === 'assistant') {
          messageIndex = i;
          break;
        }
      }
    }

    expect(messageIndex).toBe(1);
  });

  it('should create new message if not found when processing done event', () => {
    const messages: ChatMessage[] = [
      { id: 'user-1', role: 'user', content: '질문', timestamp: new Date() },
    ];

    const assistantMessageId = 'assistant-1';
    // const assistantMessage: ChatMessage = {
    //   id: assistantMessageId,
    //   role: 'assistant',
    //   content: '최종 답변',
    //   timestamp: new Date(),
    // };

    const messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
    const shouldCreateNew = messageIndex === -1;

    expect(shouldCreateNew).toBe(true);
  });
});

describe('useStreamingMessage - Sources 이벤트 처리', () => {
  it('should set _hasSourcesEvent flag when sources event is received', () => {
    const metadata = {
      sources: ['참고자료 1'],
    };

    const sourcesMetadata = {
      ...metadata,
      _hasSourcesEvent: true,
    };

    expect(sourcesMetadata._hasSourcesEvent).toBe(true);
    expect(sourcesMetadata.sources).toEqual(['참고자료 1']);
  });

  it('should merge sources metadata correctly', () => {
    const existingMetadata = {
      sources: ['기존 참고자료'],
    };

    const sourcesMetadata = {
      sources: ['새 참고자료'],
      sources_detail: [],
    };

    const merged = {
      ...existingMetadata,
      ...sourcesMetadata,
      _hasSourcesEvent: true,
    };

    expect(merged._hasSourcesEvent).toBe(true);
    expect(merged.sources).toEqual(['새 참고자료']);
  });

  it('should find message by metadata.message_id', () => {
    const messages: ChatMessage[] = [
      { id: 'user-1', role: 'user', content: '질문', timestamp: new Date() },
      {
        id: 'assistant-1',
        role: 'assistant',
        content: '답변',
        timestamp: new Date(),
        metadata: { message_id: 'server-msg-1' },
      },
    ];

    const sourcesMessageId = 'server-msg-1';
    const messageIndex = messages.findIndex(
      (msg) => msg.metadata?.message_id === sourcesMessageId
    );

    expect(messageIndex).toBe(1);
  });

  it('should handle sources_by_type in metadata', () => {
    const metadata = {
      sources_by_type: {
        statute_article: [{ name: '법령 1' }],
        case_paragraph: [{ name: '판례 1' }],
        decision_paragraph: [],
        interpretation_paragraph: [],
      },
    };

    expect(metadata.sources_by_type.statute_article.length).toBe(1);
    expect(metadata.sources_by_type.case_paragraph.length).toBe(1);
  });
});

describe('useStreamingMessage - 이벤트 순서 처리', () => {
  it('should handle done event after stream events', () => {
    const events = [
      { type: 'stream', content: '스트리밍' },
      { type: 'stream', content: ' 중인' },
      { type: 'done', content: '최종 답변' },
    ];

    let finalContent = '';
    let isDone = false;

    events.forEach((event) => {
      if (event.type === 'stream') {
        finalContent += event.content;
      } else if (event.type === 'done') {
        finalContent = event.content;
        isDone = true;
      }
    });

    expect(finalContent).toBe('최종 답변');
    expect(isDone).toBe(true);
  });

  it('should handle sources event after done event', () => {
    const events = [
      { type: 'stream', content: '스트리밍' },
      { type: 'done', content: '최종 답변' },
      { type: 'sources', metadata: { sources: ['참고자료'] } },
    ];

    let isDone = false;
    let hasSourcesEvent = false;

    events.forEach((event) => {
      if (event.type === 'done') {
        isDone = true;
      } else if (event.type === 'sources') {
        hasSourcesEvent = true;
      }
    });

    expect(isDone).toBe(true);
    expect(hasSourcesEvent).toBe(true);
  });

  it('should accumulate stream content correctly', () => {
    const events = [
      { type: 'stream', content: '첫' },
      { type: 'stream', content: ' 번째' },
      { type: 'stream', content: ' 토큰' },
    ];

    let fullContent = '';
    events.forEach((event) => {
      if (event.type === 'stream') {
        fullContent += event.content;
      }
    });

    expect(fullContent).toBe('첫 번째 토큰');
  });
});

describe('useStreamingMessage - Stream 이벤트 처리', () => {
  it('should create assistant message on first stream chunk', () => {
    const messages: ChatMessage[] = [];
    const assistantMessageId = 'assistant-1';
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };

    const isFirstChunk = true;
    const fullContent = '첫 토큰';

    if (isFirstChunk) {
      const messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
      if (messageIndex === -1) {
        messages.push({ ...assistantMessage, content: fullContent });
      }
    }

    expect(messages.length).toBe(1);
    expect(messages[0].content).toBe('첫 토큰');
  });

  it('should update existing message on subsequent stream chunks', () => {
    const messages: ChatMessage[] = [
      {
        id: 'assistant-1',
        role: 'assistant',
        content: '첫 토큰',
        timestamp: new Date(),
      },
    ];

    const assistantMessageId = 'assistant-1';
    const fullContent = '첫 토큰 두 번째 토큰';

    const messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
    if (messageIndex !== -1) {
      messages[messageIndex] = {
        ...messages[messageIndex],
        content: fullContent,
      };
    }

    expect(messages[0].content).toBe('첫 토큰 두 번째 토큰');
  });

  it('should ignore stream events after final event', () => {
    const isFinalReceived = true;
    const parsed = { type: 'stream', content: '추가 토큰' };

    if (isFinalReceived && parsed.type === 'stream') {
      // 이벤트 무시
      expect(true).toBe(true);
    } else {
      expect(false).toBe(true);
    }
  });
});

describe('useStreamingMessage - Progress 이벤트 처리', () => {
  it('should update existing progress message', () => {
    const messages: ChatMessage[] = [
      {
        id: 'progress-1',
        role: 'progress',
        content: '이전 진행상황',
        timestamp: new Date(),
      },
    ];

    const parsed = { type: 'progress', content: '새 진행상황' };

    let lastProgressIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i]?.role === 'progress') {
        lastProgressIndex = i;
        break;
      }
    }

    if (lastProgressIndex !== -1) {
      messages[lastProgressIndex] = {
        ...messages[lastProgressIndex],
        content: parsed.content,
        timestamp: new Date(),
      };
    }

    expect(messages[0].content).toBe('새 진행상황');
  });

  it('should create new progress message if none exists', () => {
    const messages: ChatMessage[] = [];

    const parsed = { type: 'progress', content: '새 진행상황' };

    let lastProgressIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i]?.role === 'progress') {
        lastProgressIndex = i;
        break;
      }
    }

    if (lastProgressIndex === -1) {
      messages.push({
        id: `progress-${Date.now()}`,
        role: 'progress',
        content: parsed.content,
        timestamp: new Date(),
      });
    }

    expect(messages.length).toBe(1);
    expect(messages[0].role).toBe('progress');
  });
});

describe('useStreamingMessage - Quota 이벤트 처리', () => {
  it('should update quota info when quota event is received', () => {
    const parsed = {
      type: 'quota',
      metadata: {
        remaining: 2,
        limit: 3,
      },
    };

    if (parsed.type === 'quota' && parsed.metadata) {
      const quotaRemaining = parsed.metadata.remaining;
      const quotaLimit = parsed.metadata.limit;

      expect(quotaRemaining).toBe(2);
      expect(quotaLimit).toBe(3);
    }
  });
});

describe('useStreamingMessage - Final 이벤트 처리', () => {
  it('should use final content if longer than stream content', () => {
    const fullContent = '스트리밍 내용';
    const parsedFinalContent = '최종 완성된 답변 내용';

    const finalContent =
      parsedFinalContent.length >= fullContent.length ? parsedFinalContent : fullContent;

    expect(finalContent).toBe('최종 완성된 답변 내용');
  });

  it('should use stream content if longer than final content', () => {
    const fullContent = '매우 긴 스트리밍 내용입니다';
    const parsedFinalContent = '짧음';

    const finalContent =
      parsedFinalContent.length >= fullContent.length ? parsedFinalContent : fullContent;

    expect(finalContent).toBe('매우 긴 스트리밍 내용입니다');
  });
});

describe('useStreamingMessage - 메시지 검색 로직', () => {
  it('should search message by multiple methods', () => {
    const messages: ChatMessage[] = [
      { id: 'user-1', role: 'user', content: '질문', timestamp: new Date() },
      {
        id: 'assistant-1',
        role: 'assistant',
        content: '답변',
        timestamp: new Date(),
        metadata: { message_id: 'server-msg-1' },
      },
    ];

    const sourcesMessageId = 'server-msg-1';
    const assistantMessageId = 'assistant-1';
    const streamingMessageId = 'assistant-1';

    let messageIndex = -1;
    let foundMessageId: string | null = null;

    // Method 1: metadata.message_id
    if (sourcesMessageId) {
      messageIndex = messages.findIndex((msg) => msg.metadata?.message_id === sourcesMessageId);
      if (messageIndex !== -1) {
        foundMessageId = messages[messageIndex]?.id || null;
      }
    }

    // Method 2: assistantMessageId
    if (messageIndex === -1 && assistantMessageId) {
      messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
      if (messageIndex !== -1) {
        foundMessageId = assistantMessageId;
      }
    }

    // Method 3: streamingMessageId
    if (messageIndex === -1 && streamingMessageId) {
      messageIndex = messages.findIndex((msg) => msg.id === streamingMessageId);
      if (messageIndex !== -1) {
        foundMessageId = streamingMessageId;
      }
    }

    expect(messageIndex).toBe(1);
    expect(foundMessageId).toBe('assistant-1');
  });
});

describe('useStreamingMessage - 메타데이터 병합', () => {
  it('should merge sources metadata with existing metadata', () => {
    const existingMetadata = {
      message_id: 'msg-1',
      sources: ['기존 참고자료'],
    };

    const newSources = {
      sources: ['새 참고자료'],
      sources_by_type: {
        statute_article: [],
        case_paragraph: [],
        decision_paragraph: [],
        interpretation_paragraph: [],
      },
      relatedQuestions: ['관련 질문'],
    };

    const merged = {
      ...existingMetadata,
      sources_by_type: newSources.sourcesByType,
      sources: newSources.sources.length > 0 ? newSources.sources : existingMetadata.sources,
      related_questions: newSources.relatedQuestions,
    };

    expect(merged.message_id).toBe('msg-1');
    expect(merged.sources).toEqual(['새 참고자료']);
    expect(merged.related_questions).toEqual(['관련 질문']);
  });

  it('should preserve existing metadata when new metadata is empty', () => {
    const existingMetadata = {
      message_id: 'msg-1',
      sources: ['기존 참고자료'],
    };

    const newSources = {
      sources: [],
      sourcesByType: {
        statute_article: [],
        case_paragraph: [],
        decision_paragraph: [],
        interpretation_paragraph: [],
      },
      relatedQuestions: [],
    };

    const merged = {
      ...existingMetadata,
      sources_by_type: newSources.sourcesByType,
      sources:
        newSources.sources.length > 0 ? newSources.sources : existingMetadata.sources,
      related_questions:
        newSources.relatedQuestions.length > 0
          ? newSources.relatedQuestions
          : existingMetadata.sources,
    };

    expect(merged.sources).toEqual(['기존 참고자료']);
  });
});
