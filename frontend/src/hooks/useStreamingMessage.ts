/**
 * 스트리밍 메시지 처리 훅
 */
import { useRef, useCallback } from 'react';
import { useChat } from './useChat';
import { parseStreamChunk } from '../utils/streamParser';
import { parseSourcesMetadata, mergeSourcesMetadata } from '../utils/sourcesParser';
import { toStreamError, extractQuotaInfo } from '../utils/errorHandler';
import logger from '../utils/logger';
import type { AxiosError } from 'axios';
import type { ChatMessage, FileAttachment } from '../types/chat';
import type { StreamError } from '../types/error';

interface MessageSearchResult {
  messageIndex: number;
  foundMessageId: string | null;
  searchMethod: string;
}

function findMessageByIdentifiers(
  messages: ChatMessage[],
  options: {
    sourcesMessageId?: string;
    assistantMessageId?: string;
    streamingMessageId?: string | null;
  }
): MessageSearchResult {
  const { sourcesMessageId, assistantMessageId, streamingMessageId } = options;
  let messageIndex = -1;
  let foundMessageId: string | null = null;
  let searchMethod = '';

  if (sourcesMessageId) {
    messageIndex = messages.findIndex((msg) => msg.metadata?.message_id === sourcesMessageId);
    if (messageIndex !== -1) {
      foundMessageId = messages[messageIndex]?.id || null;
      searchMethod = 'metadata.message_id';
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Found message by metadata.message_id:', foundMessageId);
      }
      return { messageIndex, foundMessageId, searchMethod };
    }
  }

  if (assistantMessageId) {
    messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
    if (messageIndex !== -1) {
      foundMessageId = assistantMessageId;
      searchMethod = 'assistantMessageId';
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Found message by assistantMessageId:', foundMessageId);
      }
      return { messageIndex, foundMessageId, searchMethod };
    }
  }

  if (sourcesMessageId) {
    messageIndex = messages.findIndex((msg) => msg.id === sourcesMessageId);
    if (messageIndex !== -1) {
      foundMessageId = sourcesMessageId;
      searchMethod = 'sourcesMessageId as id';
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Found message by sourcesMessageId as id:', foundMessageId);
      }
      return { messageIndex, foundMessageId, searchMethod };
    }
  }

  if (streamingMessageId) {
    messageIndex = messages.findIndex((msg) => msg.id === streamingMessageId);
    if (messageIndex !== -1) {
      foundMessageId = streamingMessageId;
      searchMethod = 'streamingMessageId';
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Found message by streamingMessageId:', foundMessageId);
      }
      return { messageIndex, foundMessageId, searchMethod };
    }
  }

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg && msg.role === 'assistant') {
      messageIndex = i;
      foundMessageId = msg.id;
      searchMethod = 'last assistant message (fallback)';
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Found message by last assistant message (fallback):', foundMessageId);
      }
      return { messageIndex, foundMessageId, searchMethod };
    }
  }

  return { messageIndex: -1, foundMessageId: null, searchMethod: '' };
}

function findMessageForDoneEvent(
  messages: ChatMessage[],
  assistantMessageId: string,
  streamingMessageId: string | null
): MessageSearchResult {
  let messageIndex = messages.findIndex((msg) => msg.id === assistantMessageId);
  let foundMessageId: string | null = assistantMessageId;
  let searchMethod = 'assistantMessageId';

  if (messageIndex === -1 && streamingMessageId) {
    messageIndex = messages.findIndex((msg) => msg.id === streamingMessageId);
    if (messageIndex !== -1) {
      foundMessageId = streamingMessageId;
      searchMethod = 'streamingMessageId';
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Done event: Found message by streamingMessageId:', foundMessageId);
      }
      return { messageIndex, foundMessageId, searchMethod };
    }
  }

  if (messageIndex === -1) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg && msg.role === 'assistant') {
        messageIndex = i;
        foundMessageId = msg.id;
        searchMethod = 'last assistant message (fallback)';
        if (import.meta.env.DEV) {
          logger.debug('[Stream] Done event: Found message by last assistant message (fallback):', foundMessageId);
        }
        return { messageIndex, foundMessageId, searchMethod };
      }
    }
  }

  return { messageIndex, foundMessageId, searchMethod };
}

function shouldClearStreamingId(
  streamingMessageId: string | null,
  assistantMessageId: string,
  foundMessageId: string | null
): boolean {
  if (!streamingMessageId) return false;
  return (
    streamingMessageId === assistantMessageId ||
    streamingMessageId === foundMessageId
  );
}

interface UseStreamingMessageOptions {
  isAuthenticated: boolean;
  streamingMessageId: string | null;
  setStreamingId: (id: string | null) => void;
  updateMessages: (updater: (messages: ChatMessage[]) => ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  addError: (messageId: string, error: StreamError) => void;
  setQuotaInfo: (info: { remaining: number; limit: number } | null) => void;
  setCurrentProgress: (progress: string | null) => void;
  setProgressHistory: (history: string[]) => void;
  setSessionListRefreshTrigger: (updater: (prev: number) => number) => void;
  showToast: (toast: { message: string; type: 'error' | 'success' | 'warning'; action?: { label: string; onClick: () => void } }) => void;
  login: () => void;
  streamErrors: Map<string, StreamError>;
}

export function useStreamingMessage(options: UseStreamingMessageOptions) {
  const { sendStreamingMessage } = useChat();
  const tokenBufferRef = useRef<Map<string, string>>(new Map());
  const tokenBufferTimeoutRef = useRef<Map<string, number>>(new Map());
  const sourcesWaitTimeoutRef = useRef<Map<string, number>>(new Map());

  const handleStreamingMessage = useCallback(
    async (
      message: string,
      sessionId: string,
      attachments?: FileAttachment[],
      imageBase64?: string,
      fileBase64?: string,
      filename?: string,
      skipUserMessage: boolean = false
    ) => {
      const {
        isAuthenticated,
        streamingMessageId,
        setStreamingId,
        updateMessages,
        addMessage,
        addError,
        setQuotaInfo,
        setCurrentProgress,
        setProgressHistory,
        setSessionListRefreshTrigger,
        showToast,
        login,
        streamErrors,
      } = options;

      if (!skipUserMessage) {
        const userMessage: ChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: message,
          timestamp: new Date(),
          attachments,
        };
        addMessage(userMessage);
      }

      const assistantMessageId = `assistant-${Date.now()}`;
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      };

      let isFirstChunk = true;
      let isFinalReceived = false;
      let fullContent = '';

      try {
        setCurrentProgress(null);
        setProgressHistory([]);

        await sendStreamingMessage(
          message,
          sessionId,
          (chunk) => {
            if (import.meta.env.DEV) {
              logger.debug('[Stream] Chunk received:', JSON.stringify(chunk));
            }

            const parsed = parseStreamChunk(chunk);

            if (parsed.type === 'quota' && parsed.metadata) {
              const quotaRemaining = parsed.metadata.remaining;
              const quotaLimit = parsed.metadata.limit;
              if (quotaRemaining !== undefined && quotaLimit !== undefined) {
                setQuotaInfo({ remaining: quotaRemaining, limit: quotaLimit });
              }
              return;
            }

            if (parsed.type === 'progress') {
              updateMessages((prev) => {
                let lastProgressIndex = -1;
                for (let i = prev.length - 1; i >= 0; i--) {
                  const msg = prev[i];
                  if (msg && msg.role === 'progress') {
                    lastProgressIndex = i;
                    break;
                  }
                }

                if (lastProgressIndex !== -1) {
                  const updated = [...prev];
                  const existingMsg = updated[lastProgressIndex];
                  if (existingMsg) {
                    updated[lastProgressIndex] = {
                      ...existingMsg,
                      content: parsed.content,
                      timestamp: new Date(),
                      metadata: parsed.metadata ? {
                        ...existingMsg.metadata,
                        ...parsed.metadata,
                      } : existingMsg.metadata,
                    };
                  }
                  if (import.meta.env.DEV) {
                    logger.debug('[Stream] Progress message updated:', parsed.content);
                  }
                  return updated;
                } else {
                  const progressMessage: ChatMessage = {
                    id: `progress-${Date.now()}`,
                    role: 'progress',
                    content: parsed.content,
                    timestamp: new Date(),
                    metadata: parsed.metadata ? {
                      ...parsed.metadata,
                    } : undefined,
                  };
                  if (import.meta.env.DEV) {
                    logger.debug('[Stream] Progress message added:', parsed.content);
                  }
                  return [...prev, progressMessage];
                }
              });

              setCurrentProgress(parsed.content);
            } else if (parsed.type === 'stream') {
              if (isFinalReceived) {
                if (import.meta.env.DEV) {
                  logger.debug('[Stream] Ignoring stream event after final:', parsed.content.substring(0, 50));
                }
                return;
              }

              fullContent += parsed.content;
              if (import.meta.env.DEV) {
                logger.debug('[Stream] Token received, full content so far length:', fullContent.length);
              }

              if (isFirstChunk) {
                setStreamingId(assistantMessageId);
                if (import.meta.env.DEV) {
                  logger.debug('[Stream] First chunk received, setting streamingMessageId:', assistantMessageId);
                }
              }

              updateMessages((prevMessages) => {
                const messageIndex = prevMessages.findIndex((msg) => msg.id === assistantMessageId);

                if (isFirstChunk && messageIndex === -1) {
                  if (import.meta.env.DEV) {
                    logger.debug('[Stream] First chunk received, adding assistant message:', assistantMessageId, 'Content length:', fullContent.length);
                  }
                  return [...prevMessages, { ...assistantMessage, content: fullContent }];
                }

                if (messageIndex !== -1) {
                  const updated = [...prevMessages];
                  const existingMsg = updated[messageIndex];
                  if (existingMsg) {
                    updated[messageIndex] = {
                      ...existingMsg,
                      content: fullContent,
                    };
                  }
                  if (import.meta.env.DEV && fullContent.length % 50 === 0) {
                    logger.debug('[Stream] Message updated at index:', messageIndex, 'Content length:', fullContent.length);
                  }
                  return updated;
                }

                if (import.meta.env.DEV) {
                  logger.warn('[Stream] Message not found, adding as fallback:', assistantMessageId);
                }
                return [...prevMessages, { ...assistantMessage, content: fullContent }];
              });

              if (isFirstChunk) {
                isFirstChunk = false;
              }
            } else if (parsed.type === 'chunk') {
              fullContent = parsed.content;

              if (isFirstChunk) {
                setStreamingId(assistantMessageId);
                if (import.meta.env.DEV) {
                  logger.debug('[Stream] First chunk received, setting streamingMessageId:', assistantMessageId);
                }
              }

              updateMessages((prevMessages) => {
                const messageIndex = prevMessages.findIndex((msg) => msg.id === assistantMessageId);

                if (isFirstChunk && messageIndex === -1) {
                  return [...prevMessages, {
                    ...assistantMessage,
                    content: parsed.content,
                    metadata: {
                      message_id: parsed.metadata?.message_id,
                    },
                  }];
                } else if (messageIndex !== -1) {
                  const updated = [...prevMessages];
                  const existingMsg = updated[messageIndex];
                  if (existingMsg) {
                    updated[messageIndex] = {
                      ...existingMsg,
                      content: parsed.content,
                      metadata: {
                        ...existingMsg.metadata,
                        message_id: parsed.metadata?.message_id || existingMsg.metadata?.message_id,
                      },
                    };
                  }
                  return updated;
                }
                return prevMessages;
              });

              if (isFirstChunk) {
                isFirstChunk = false;
              }
            } else if (parsed.type === 'sources') {
              if (parsed.metadata) {
                const sourcesMetadata = parsed.metadata;
                const sourcesMessageId = sourcesMetadata.message_id as string | undefined;

                if (import.meta.env.DEV) {
                  logger.debug('[Stream] Sources event received:', {
                    messageId: sourcesMessageId,
                    assistantMessageId,
                    sourcesByType: sourcesMetadata.sources_by_type,
                    sources: sourcesMetadata.sources,
                    legalReferences: sourcesMetadata.legal_references,
                    sourcesDetail: sourcesMetadata.sources_detail,
                    metadataKeys: Object.keys(sourcesMetadata),
                  });
                }

                let messageFound = false;
                let searchResult: MessageSearchResult = {
                  messageIndex: -1,
                  foundMessageId: null,
                  searchMethod: '',
                };
                updateMessages((prev) => {
                  const result = findMessageByIdentifiers(prev, {
                    sourcesMessageId,
                    assistantMessageId,
                    streamingMessageId,
                  });
                  searchResult = result;
                  const { messageIndex, foundMessageId, searchMethod } = result;

                  if (messageIndex === -1) {
                    if (import.meta.env.DEV) {
                      logger.warn('[Stream] Sources event: Message not found with all search methods', {
                        sourcesMessageId,
                        assistantMessageId,
                        streamingMessageId,
                        availableMessages: prev.map(msg => ({
                          id: msg.id,
                          role: msg.role,
                          metadataMessageId: msg.metadata?.message_id,
                          isAssistant: msg.role === 'assistant',
                        })),
                      });
                    }
                  } else if (import.meta.env.DEV) {
                    logger.debug('[Stream] Message found successfully:', {
                      searchMethod,
                      messageIndex,
                      foundMessageId,
                      sourcesMessageId,
                      assistantMessageId,
                      streamingMessageId,
                    });
                  }

                  if (messageIndex !== -1) {
                    messageFound = true;
                    const updated = [...prev];
                    const existingMsg = updated[messageIndex];
                    if (existingMsg) {
                      const sourcesCount = Array.isArray(sourcesMetadata.sources) ? sourcesMetadata.sources.length : 0;
                      const sourcesDetailCount = Array.isArray(sourcesMetadata.sources_detail) ? sourcesMetadata.sources_detail.length : 0;

                      if (sourcesCount > 0 && sourcesDetailCount > 0 && sourcesCount !== sourcesDetailCount) {
                        if (import.meta.env.DEV) {
                          logger.warn('[Stream] Sources count mismatch:', {
                            sourcesCount,
                            sourcesDetailCount,
                            sources: sourcesMetadata.sources,
                            sourcesDetail: sourcesMetadata.sources_detail,
                          });
                        }
                      }

                      let finalSourcesDetail: unknown = sourcesMetadata.sources_detail;
                      if (sourcesCount > 0 && sourcesDetailCount === 0 && Array.isArray(sourcesMetadata.sources)) {
                        finalSourcesDetail = sourcesMetadata.sources.map((source) => ({
                          name: source,
                          type: 'case_paragraph',
                          url: '',
                          metadata: {},
                          content: source,
                        }));
                        if (import.meta.env.DEV) {
                          logger.debug('[Stream] Generated sources_detail from sources array:', {
                            generatedCount: Array.isArray(finalSourcesDetail) ? finalSourcesDetail.length : 0,
                          });
                        }
                      }

                      const parsedSources = parseSourcesMetadata({
                        ...sourcesMetadata,
                        sources_detail: finalSourcesDetail,
                        message_id: sourcesMessageId,
                      });

                      if (import.meta.env.DEV) {
                        logger.debug('[Stream] Parsed sources:', {
                          sourcesByType: parsedSources.sourcesByType,
                          sourcesCount: parsedSources.sources.length,
                          legalReferencesCount: parsedSources.legalReferences.length,
                          sourcesDetailCount: parsedSources.sourcesDetail.length,
                          relatedQuestionsCount: parsedSources.relatedQuestions.length,
                          sourcesCountMatch: parsedSources.sources.length === parsedSources.sourcesDetail.length,
                          parsedSources,
                        });
                      }

                      const mergedMetadata = mergeSourcesMetadata(existingMsg.metadata, parsedSources);

                      // sources 이벤트 플래그 추가
                      const finalMetadata = {
                        ...mergedMetadata,
                        _hasSourcesEvent: true,
                      };

                      // 메시지 객체를 깊은 복사하여 React가 변경을 감지하도록 함
                      updated[messageIndex] = {
                        ...existingMsg,
                        metadata: { ...finalMetadata },
                      };

                      if (import.meta.env.DEV) {
                        logger.debug('[Stream] Message metadata updated with sources:', {
                          messageIndex,
                          messageId: existingMsg?.id,
                          metadataMessageId: existingMsg?.metadata?.message_id,
                          sourcesMessageId,
                          sourcesByType: parsedSources.sourcesByType,
                          sourcesCount: parsedSources.sources.length,
                          legalReferencesCount: parsedSources.legalReferences.length,
                          sourcesDetailCount: parsedSources.sourcesDetail.length,
                          relatedQuestionsCount: parsedSources.relatedQuestions.length,
                          hasRelatedQuestions: parsedSources.relatedQuestions.length > 0,
                          mergedMetadata,
                          finalMetadata: updated[messageIndex]?.metadata,
                          metadataKeys: Object.keys(mergedMetadata),
                          hasSourcesByTypeInMetadata: !!mergedMetadata.sources_by_type,
                          hasSourcesInMetadata: Array.isArray(mergedMetadata.sources) && mergedMetadata.sources.length > 0,
                          hasSourcesDetailInMetadata: Array.isArray(mergedMetadata.sources_detail) && mergedMetadata.sources_detail.length > 0,
                          hasRelatedQuestionsInMetadata: Array.isArray(mergedMetadata.related_questions) && mergedMetadata.related_questions.length > 0,
                        });
                      }
                    }

                    return updated;
                  } else {
                    if (import.meta.env.DEV) {
                      logger.warn('[Stream] Sources event: Message not found, attempting to add new message', {
                        sourcesMessageId,
                        assistantMessageId,
                        messageIds: prev.map(msg => ({ id: msg.id, role: msg.role, metadataMessageId: msg.metadata?.message_id })),
                      });
                    }

                    const lastAssistantIndex = prev.length - 1;
                    if (lastAssistantIndex >= 0 && prev[lastAssistantIndex]?.role === 'assistant') {
                      messageFound = true;
                      const updated = [...prev];
                      const lastMsg = updated[lastAssistantIndex];
                      if (lastMsg) {
                        const sourcesCount = Array.isArray(sourcesMetadata.sources) ? sourcesMetadata.sources.length : 0;
                        const sourcesDetailCount = Array.isArray(sourcesMetadata.sources_detail) ? sourcesMetadata.sources_detail.length : 0;

                        if (sourcesCount > 0 && sourcesDetailCount > 0 && sourcesCount !== sourcesDetailCount) {
                          if (import.meta.env.DEV) {
                            logger.warn('[Stream] Sources count mismatch (fallback):', {
                              sourcesCount,
                              sourcesDetailCount,
                              sources: sourcesMetadata.sources,
                              sourcesDetail: sourcesMetadata.sources_detail,
                            });
                          }
                        }

                        let finalSourcesDetail: unknown = sourcesMetadata.sources_detail;
                        if (sourcesCount > 0 && sourcesDetailCount === 0 && Array.isArray(sourcesMetadata.sources)) {
                          finalSourcesDetail = sourcesMetadata.sources.map((source) => ({
                            name: source,
                            type: 'case_paragraph',
                            url: '',
                            metadata: {},
                            content: source,
                          }));
                          if (import.meta.env.DEV) {
                            logger.debug('[Stream] Generated sources_detail from sources array (fallback):', {
                              generatedCount: Array.isArray(finalSourcesDetail) ? finalSourcesDetail.length : 0,
                            });
                          }
                        }

                        const parsedSources = parseSourcesMetadata({
                          ...sourcesMetadata,
                          sources_detail: finalSourcesDetail,
                          message_id: sourcesMessageId,
                        });

                        if (import.meta.env.DEV) {
                          logger.debug('[Stream] Parsed sources (fallback):', {
                            sourcesCount: parsedSources.sources.length,
                            legalReferencesCount: parsedSources.legalReferences.length,
                            sourcesDetailCount: parsedSources.sourcesDetail.length,
                            relatedQuestionsCount: parsedSources.relatedQuestions.length,
                            parsedSources,
                          });
                        }

                        const mergedMetadata = mergeSourcesMetadata(lastMsg.metadata, parsedSources);

                        updated[lastAssistantIndex] = {
                          ...lastMsg,
                          metadata: mergedMetadata,
                        };

                        if (import.meta.env.DEV) {
                          logger.debug('[Stream] Updated last assistant message with sources (fallback):', {
                            messageId: lastMsg.id,
                            sourcesCount: parsedSources.sources.length,
                            legalReferencesCount: parsedSources.legalReferences.length,
                            sourcesDetailCount: parsedSources.sourcesDetail.length,
                            relatedQuestionsCount: parsedSources.relatedQuestions.length,
                            mergedMetadata,
                            finalMetadata: updated[lastAssistantIndex]?.metadata,
                            metadataKeys: Object.keys(mergedMetadata),
                            hasSourcesInMetadata: Array.isArray(mergedMetadata.sources) && mergedMetadata.sources.length > 0,
                            hasSourcesDetailInMetadata: Array.isArray(mergedMetadata.sources_detail) && mergedMetadata.sources_detail.length > 0,
                            hasRelatedQuestionsInMetadata: Array.isArray(mergedMetadata.related_questions) && mergedMetadata.related_questions.length > 0,
                          });
                        }

                        return updated;
                      }
                    }
                  }
                  return prev;
                });

                if (messageFound && searchResult) {
                  const foundId = searchResult.foundMessageId;
                  if (shouldClearStreamingId(streamingMessageId, assistantMessageId, foundId) ||
                    (sourcesMessageId && streamingMessageId === sourcesMessageId) ||
                    !streamingMessageId) {
                    setStreamingId(null);
                    if (import.meta.env.DEV) {
                      logger.debug('[Stream] StreamingMessageId cleared after sources event');
                    }
                  }
                } else if (streamingMessageId !== null) {
                  if (import.meta.env.DEV) {
                    logger.warn('[Stream] Message not found but clearing streamingMessageId:', {
                      wasStreamingMessageId: streamingMessageId,
                      sourcesMessageId,
                      assistantMessageId,
                    });
                  }
                  setStreamingId(null);
                }

                const timeoutToCancel = sourcesWaitTimeoutRef.current.get(assistantMessageId) ||
                  sourcesWaitTimeoutRef.current.get(sourcesMessageId || '');
                if (timeoutToCancel) {
                  clearTimeout(timeoutToCancel);
                  if (sourcesWaitTimeoutRef.current.has(assistantMessageId)) {
                    sourcesWaitTimeoutRef.current.delete(assistantMessageId);
                  }
                  if (sourcesMessageId && sourcesWaitTimeoutRef.current.has(sourcesMessageId)) {
                    sourcesWaitTimeoutRef.current.delete(sourcesMessageId);
                  }
                  if (import.meta.env.DEV) {
                    logger.debug('[Stream] Sources wait timeout cancelled (sources event received)', {
                      assistantMessageId,
                      sourcesMessageId,
                    });
                  }
                }
              }
            } else if (parsed.type === 'done') {
              const finalContent = parsed.content && parsed.content.trim() ? parsed.content : fullContent;

              if (import.meta.env.DEV) {
                logger.debug('[Stream] Done event received, final content length:', finalContent.length);
                if (parsed.metadata) {
                  logger.debug('[Stream] Done metadata:', parsed.metadata);
                }
              }

              tokenBufferRef.current.delete(assistantMessageId);
              if (tokenBufferTimeoutRef.current.has(assistantMessageId)) {
                clearTimeout(tokenBufferTimeoutRef.current.get(assistantMessageId)!);
                tokenBufferTimeoutRef.current.delete(assistantMessageId);
              }

              const foundMessageIdRef = { value: assistantMessageId };
              updateMessages((prev) => {
                const searchResult = findMessageForDoneEvent(prev, assistantMessageId, streamingMessageId);
                const { messageIndex, foundMessageId, searchMethod } = searchResult;
                foundMessageIdRef.value = foundMessageId || assistantMessageId;

                if (messageIndex !== -1) {
                  const updated = [...prev];
                  const existingMsg = updated[messageIndex];
                  if (existingMsg) {
                    let mergedMetadata = existingMsg.metadata || {};
                    if (parsed.metadata) {
                      // done 이벤트의 metadata를 파싱 (sources_by_type 우선 사용)
                      const parsedSources = parseSourcesMetadata(parsed.metadata, {
                        skipSourcesDetailReconstruction: false,  // sourcesDetail도 재구성하여 하위 호환성 보장
                      });

                      if (import.meta.env.DEV) {
                        logger.debug('[Stream] Done event parsed sources:', {
                          messageId: foundMessageId,
                          searchMethod,
                          sourcesCount: parsedSources.sources.length,
                          legalReferencesCount: parsedSources.legalReferences.length,
                          sourcesDetailCount: parsedSources.sourcesDetail.length,
                          relatedQuestionsCount: parsedSources.relatedQuestions.length,
                          sourcesByTypeKeys: Object.keys(parsedSources.sourcesByType),
                          sourcesByTypeCounts: {
                            statute: parsedSources.sourcesByType.statute_article.length,
                            case: parsedSources.sourcesByType.case_paragraph.length,
                            decision: parsedSources.sourcesByType.decision_paragraph.length,
                            interpretation: parsedSources.sourcesByType.interpretation_paragraph.length,
                          },
                          parsedSources,
                        });
                      }

                      mergedMetadata = mergeSourcesMetadata(existingMsg.metadata, parsedSources);
                    } else {
                      if (import.meta.env.DEV) {
                        logger.warn('[Stream] Done event received but no metadata in parsed data:', {
                          messageId: foundMessageId,
                          searchMethod,
                          parsed,
                        });
                      }
                    }

                    mergedMetadata = {
                      ...mergedMetadata,
                      _isDone: true,
                    };

                    if (import.meta.env.DEV) {
                      logger.debug('[Stream] Done event merged metadata:', {
                        messageId: foundMessageId,
                        searchMethod,
                        mergedMetadata,
                        metadataKeys: Object.keys(mergedMetadata),
                      });
                    }

                    // done 이벤트의 content로 답변 업데이트
                    // 메시지 객체를 깊은 복사하여 React가 변경을 감지하도록 함
                    updated[messageIndex] = {
                      ...existingMsg,
                      content: finalContent,
                      metadata: { ...mergedMetadata }, // 새로운 객체로 복사
                    };
                  }
                  return updated;
                } else {
                  // 메시지를 찾지 못한 경우 새로 생성
                  if (import.meta.env.DEV) {
                    logger.warn('[Stream] Done event: Message not found, creating new message:', {
                      assistantMessageId,
                      streamingMessageId,
                      availableMessages: prev.map(msg => ({
                        id: msg.id,
                        role: msg.role,
                        isAssistant: msg.role === 'assistant',
                      })),
                    });
                  }

                  const parsedSources = parsed.metadata
                    ? parseSourcesMetadata(parsed.metadata, {
                        skipSourcesDetailReconstruction: false,
                      })
                    : {
                        sources: [],
                        legalReferences: [],
                        sourcesDetail: [],
                        sourcesByType: {
                          statute_article: [],
                          case_paragraph: [],
                          decision_paragraph: [],
                          interpretation_paragraph: [],
                        },
                        relatedQuestions: [],
                        messageId: undefined,
                      };

                  return [...prev, {
                    ...assistantMessage,
                    content: finalContent,
                    metadata: {
                      ...mergeSourcesMetadata(undefined, parsedSources),
                      _isDone: true,
                    },
                  }];
                }
              });

              const finalFoundMessageId = foundMessageIdRef.value;
              if (shouldClearStreamingId(streamingMessageId, assistantMessageId, finalFoundMessageId)) {
                setStreamingId(null);
                if (import.meta.env.DEV) {
                  logger.debug('[Stream] Done event: stopping typing effect, streamingMessageId set to null', {
                    assistantMessageId,
                    streamingMessageId,
                    foundMessageId: finalFoundMessageId,
                  });
                }
              }

              setSessionListRefreshTrigger(prev => prev + 1);
            } else if (parsed.metadata && 'sources' in parsed.metadata) {
              if (parsed.metadata) {
                const sourcesMetadata = parsed.metadata;
                const sourcesMessageId = sourcesMetadata.message_id;
                const relatedQuestions = sourcesMetadata.related_questions as string[] | undefined;

                if (import.meta.env.DEV) {
                  logger.debug('[Stream] Sources in final metadata:', {
                    messageId: sourcesMessageId,
                    sourcesByType: sourcesMetadata.sources_by_type,
                    sources: sourcesMetadata.sources,
                    legalReferences: sourcesMetadata.legal_references,
                    sourcesDetail: sourcesMetadata.sources_detail,
                    relatedQuestions: relatedQuestions,
                  });
                }

                const targetMessageId = sourcesMessageId || assistantMessageId;

                updateMessages((prev) => {
                  const messageIndex = prev.findIndex((msg) =>
                    msg.id === targetMessageId ||
                    msg.metadata?.message_id === sourcesMessageId
                  );

                  if (messageIndex !== -1) {
                    const updated = [...prev];
                    const existingMsg = updated[messageIndex];
                    if (existingMsg) {
                      const parsedSources = parseSourcesMetadata({
                        ...sourcesMetadata,
                        message_id: sourcesMessageId,
                        related_questions: Array.isArray(relatedQuestions) ? relatedQuestions : existingMsg.metadata?.related_questions,
                      }, {
                        skipSourcesDetailReconstruction: true,  // sources_by_type 직접 사용
                      });

                      updated[messageIndex] = {
                        ...existingMsg,
                        metadata: mergeSourcesMetadata(existingMsg.metadata, parsedSources),
                      };
                    }

                    if (import.meta.env.DEV && existingMsg) {
                      logger.debug('[Stream] Message metadata updated with sources from final:', {
                        messageId: targetMessageId,
                        updatedMetadata: updated[messageIndex]?.metadata,
                      });
                    }

                    return updated;
                  }
                  return prev;
                });
              }
            } else if (parsed.type === 'final') {
              isFinalReceived = true;

              const parsedFinalContent = parsed.content && parsed.content.trim() ? parsed.content : '';

              const finalContent = parsedFinalContent.length >= fullContent.length
                ? parsedFinalContent
                : fullContent;

              if (import.meta.env.DEV) {
                logger.debug('[Stream] Final event received, replacing content with final:', {
                  finalContentLength: finalContent.length,
                  parsedContentLength: parsed.content?.length || 0,
                  streamContentLength: fullContent.length,
                  previousContentLength: fullContent.length,
                  hasMetadata: !!parsed.metadata,
                  usingFinal: finalContent === parsedFinalContent,
                  usingStream: finalContent === fullContent,
                });
                if (parsed.metadata) {
                  logger.debug('[Stream] Final metadata:', parsed.metadata);
                }
              }

              if (streamingMessageId === assistantMessageId) {
                setStreamingId(null);
                if (import.meta.env.DEV) {
                  logger.debug('[Stream] Final event: stopping typing effect immediately, streamingMessageId set to null');
                }
              }

              tokenBufferRef.current.delete(assistantMessageId);
              if (tokenBufferTimeoutRef.current.has(assistantMessageId)) {
                clearTimeout(tokenBufferTimeoutRef.current.get(assistantMessageId)!);
                tokenBufferTimeoutRef.current.delete(assistantMessageId);
              }

              fullContent = finalContent;

              updateMessages((prev) => {
                const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);

                if (messageIndex !== -1) {
                  const updated = [...prev];
                  const existingMsg = updated[messageIndex];
                  if (existingMsg) {
                    let mergedMetadata = existingMsg.metadata;
                    if (parsed.metadata) {
                      const parsedSources = parseSourcesMetadata(parsed.metadata, {
                        skipSourcesDetailReconstruction: true,  // sources_by_type 직접 사용
                      });
                      mergedMetadata = mergeSourcesMetadata(existingMsg.metadata, parsedSources);

                      if (import.meta.env.DEV) {
                        logger.debug('[Stream] Final event parsed sources:', {
                          sourcesCount: parsedSources.sources.length,
                          legalReferencesCount: parsedSources.legalReferences.length,
                          sourcesDetailCount: parsedSources.sourcesDetail.length,
                          relatedQuestionsCount: parsedSources.relatedQuestions.length,
                          parsedSources,
                        });
                      }
                    } else {
                      if (import.meta.env.DEV) {
                        logger.debug('[Stream] Final event: No metadata, keeping existing metadata:', {
                          messageId: existingMsg.id,
                          existingMetadata: existingMsg.metadata,
                        });
                      }
                    }

                    // 메시지 객체를 깊은 복사하여 React가 변경을 감지하도록 함
                    updated[messageIndex] = {
                      ...existingMsg,
                      content: finalContent,
                      metadata: mergedMetadata ? { ...mergedMetadata } : mergedMetadata,
                    };

                    if (import.meta.env.DEV) {
                      logger.debug('[Stream] Final event merged metadata:', {
                        messageId: existingMsg.id,
                        mergedMetadata,
                      });
                    }
                  }
                  return updated;
                } else {
                  const parsedSources = parsed.metadata ? parseSourcesMetadata(parsed.metadata) : {
                    sources: [],
                    legalReferences: [],
                    sourcesDetail: [],
                    sourcesByType: {
                      statute_article: [],
                      case_paragraph: [],
                      decision_paragraph: [],
                      interpretation_paragraph: [],
                    },
                    relatedQuestions: [],
                    messageId: undefined,
                  };
                  return [...prev, {
                    ...assistantMessage,
                    content: finalContent,
                    metadata: mergeSourcesMetadata(undefined, parsedSources),
                  }];
                }
              });

              setSessionListRefreshTrigger(prev => prev + 1);

              if (sourcesWaitTimeoutRef.current.has(assistantMessageId)) {
                clearTimeout(sourcesWaitTimeoutRef.current.get(assistantMessageId)!);
                sourcesWaitTimeoutRef.current.delete(assistantMessageId);
              }

              const sourcesWaitTimeout = setTimeout(() => {
                updateMessages((prev) => {
                  const currentMessage = prev.find(msg => msg.id === assistantMessageId);
                  const currentMetadata = currentMessage?.metadata;
                  const actualMessageId = currentMetadata?.message_id || assistantMessageId;

                  const hasSources = currentMetadata && Array.isArray(currentMetadata.sources) && currentMetadata.sources.length > 0;
                  const hasLegalReferences = currentMetadata && Array.isArray(currentMetadata.legal_references) && currentMetadata.legal_references.length > 0;
                  const hasSourcesDetail = currentMetadata && Array.isArray(currentMetadata.sources_detail) && currentMetadata.sources_detail.length > 0;

                  if (import.meta.env.DEV) {
                    logger.debug('[App] Done event sources check:', {
                      sessionId,
                      messageId: actualMessageId,
                      hasSources,
                      hasLegalReferences,
                      hasSourcesDetail,
                      currentMetadata,
                    });
                  }

                  if (!hasSources && !hasLegalReferences && !hasSourcesDetail) {
                    if (import.meta.env.DEV) {
                      logger.debug('[App] Sources are empty, fetching from API...', {
                        sessionId,
                        messageId: actualMessageId,
                      });
                    }

                    import('../services/chatService').then(({ getChatSources }) => {
                      getChatSources(sessionId, actualMessageId || undefined)
                        .then((sourcesData) => {
                          if (import.meta.env.DEV) {
                            logger.debug('[App] Sources fetched from API:', {
                              sessionId,
                              messageId: actualMessageId,
                              sourcesData,
                            });
                          }

                          if (sourcesData.sources.length > 0 || sourcesData.legal_references.length > 0 || sourcesData.sources_detail.length > 0) {
                            const parsedSources = parseSourcesMetadata({
                              sources: sourcesData.sources,
                              legal_references: sourcesData.legal_references,
                              sources_detail: sourcesData.sources_detail,
                            });

                            updateMessages((prev) => {
                              const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);

                              if (messageIndex !== -1) {
                                const updated = [...prev];
                                const existingMsg = updated[messageIndex];
                                if (existingMsg) {
                                  updated[messageIndex] = {
                                    ...existingMsg,
                                    metadata: mergeSourcesMetadata(existingMsg.metadata, parsedSources),
                                  };
                                }

                                if (import.meta.env.DEV) {
                                  logger.debug('[App] Message metadata updated with sources:', {
                                    messageId: assistantMessageId,
                                    updatedMetadata: updated[messageIndex]?.metadata,
                                  });
                                }

                                return updated;
                              }
                              return prev;
                            });
                          } else {
                            if (import.meta.env.DEV) {
                              logger.warn('[App] Sources fetched but empty:', {
                                sessionId,
                                messageId: actualMessageId,
                                sourcesData,
                              });
                            }
                          }
                        })
                        .catch((error) => {
                          if (import.meta.env.DEV) {
                            logger.error('[App] Error fetching sources:', error);
                          }
                        });
                    });
                  } else {
                    if (import.meta.env.DEV) {
                      logger.debug('[App] Sources already present in metadata:', {
                        sessionId,
                        messageId: actualMessageId,
                        sources: currentMetadata?.sources,
                        legalReferences: currentMetadata?.legal_references,
                        sourcesDetail: currentMetadata?.sources_detail,
                      });
                    }
                  }

                  return prev;
                });

                sourcesWaitTimeoutRef.current.delete(assistantMessageId);
              }, 1000);

              sourcesWaitTimeoutRef.current.set(assistantMessageId, sourcesWaitTimeout);

              if (import.meta.env.DEV) {
                logger.debug('[App] Done event: Waiting for sources event (timeout: 1s)', {
                  assistantMessageId,
                  timeoutId: sourcesWaitTimeout,
                });
              }
            }
          },
          imageBase64,
          fileBase64,
          filename
        );

        setCurrentProgress(null);
        if (import.meta.env.DEV) {
          logger.debug('[Stream] Streaming completed. Final content length:', fullContent.length);
        }
      } catch (error) {
        logger.error('[Stream] Streaming error:', error);

        if (streamingMessageId === assistantMessageId || streamingMessageId !== null) {
          if (import.meta.env.DEV) {
            logger.debug('[Stream] StreamingMessageId set to null due to error, was:', streamingMessageId);
          }
          setStreamingId(null);
        }

        if (fullContent.trim()) {
          updateMessages((prev) => {
            const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);

            if (messageIndex !== -1) {
              const updated = [...prev];
              const existingMsg = updated[messageIndex];
              if (existingMsg) {
                updated[messageIndex] = {
                  ...existingMsg,
                  content: fullContent,
                };
              }
              return updated;
            }
            return prev;
          });
        }

        if (error && typeof error === 'object' && 'status' in error) {
          const apiError = error as AxiosError<{ quotaInfo?: { remaining: number; limit: number } }> & { status?: number; quotaInfo?: { remaining: number; limit: number } };
          if (apiError.status === 429 && !isAuthenticated) {
            if (apiError.quotaInfo) {
              setQuotaInfo(apiError.quotaInfo);
              logger.debug('[Stream] Quota info updated from error:', apiError.quotaInfo);
            } else if (apiError.response?.headers) {
              const quotaRemaining = apiError.response.headers['x-quota-remaining'];
              const quotaLimit = apiError.response.headers['x-quota-limit'];
              if (quotaRemaining !== undefined && quotaLimit !== undefined) {
                setQuotaInfo({
                  remaining: parseInt(quotaRemaining, 10),
                  limit: parseInt(quotaLimit, 10)
                });
                logger.debug('[Stream] Quota info updated from headers:', { remaining: quotaRemaining, limit: quotaLimit });
              } else {
                setQuotaInfo({ remaining: 0, limit: 3 });
                logger.debug('[Stream] Quota info set to default (0, 3)');
              }
            } else {
              setQuotaInfo({ remaining: 0, limit: 3 });
              logger.debug('[Stream] Quota info set to default (0, 3)');
            }
          }
        } else if (error instanceof Error && error.message.includes('429') && !isAuthenticated) {
          setQuotaInfo({ remaining: 0, limit: 3 });
          logger.debug('[Stream] Quota info set to default (0, 3) from error message');
        }

        const remainingTokens = tokenBufferRef.current.get(assistantMessageId) || '';
        if (remainingTokens) {
          fullContent += remainingTokens;
          tokenBufferRef.current.delete(assistantMessageId);
        }

        if (tokenBufferTimeoutRef.current.has(assistantMessageId)) {
          clearTimeout(tokenBufferTimeoutRef.current.get(assistantMessageId)!);
          tokenBufferTimeoutRef.current.delete(assistantMessageId);
        }

        setCurrentProgress(null);

        let errorMessage = error instanceof Error ? error.message : String(error);

        if (error && typeof error === 'object' && 'response' in error) {
          const axiosError = error as AxiosError<{ detail?: string }>;
          if (axiosError.response?.data?.detail) {
            errorMessage = axiosError.response.data.detail;
          }
        }

        if (isAuthenticated && (
          errorMessage.includes('무료 질의 3회를 모두 사용하셨습니다') ||
          errorMessage.includes('무료로 3회 체험')
        )) {
          errorMessage = '요청이 너무 많습니다. 잠시 후 다시 시도하세요.';
        }

        const streamError = toStreamError(error, isAuthenticated);

        if (fullContent || error) {
          updateMessages((prev) => {
            const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
            const errorContent = fullContent || streamError.message;

            if (messageIndex === -1) {
              const errorMessage: ChatMessage = {
                ...assistantMessage,
                content: errorContent,
                metadata: {
                  ...assistantMessage.metadata,
                  error: true,
                  error_type: streamError.type,
                },
              };
              return [...prev, errorMessage];
            }
            const updated = [...prev];
            const existingMsg = updated[messageIndex];
            if (existingMsg) {
              updated[messageIndex] = {
                ...existingMsg,
                content: errorContent,
                metadata: {
                  ...existingMsg.metadata,
                  error: true,
                  error_type: streamError.type,
                },
              };
            }
            return updated;
          });
        }

        addError(assistantMessageId, streamError);

        if (!isAuthenticated || !streamError.message.includes('무료 질의 3회를 모두 사용하셨습니다')) {
          showToast({
            message: streamError.message,
            type: 'error',
            action: streamError.canRetry ? {
              label: '다시 시도',
              onClick: () => {
                const error = streamErrors.get(assistantMessageId);
                if (error) {
                  addError(assistantMessageId, {
                    ...error,
                    retryCount: (error.retryCount || 0) + 1,
                  });
                }
                handleStreamingMessage(message, sessionId, attachments, undefined, undefined, undefined, true);
              }
            } : undefined,
          });
        }
      }
    },
    [options, sendStreamingMessage]
  );

  return {
    handleStreamingMessage,
  };
}

