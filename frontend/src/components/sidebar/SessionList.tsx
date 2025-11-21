/**
 * 세션 리스트 컴포넌트 (통합 스크롤 + 자동 펼치기)
 */
import { Folder, ChevronDown, ChevronRight, Inbox } from 'lucide-react';
import { useState, useEffect, useRef, useCallback } from 'react';
import { DateGroup } from '../../utils/dateUtils';
import { SessionItem } from './SessionItem';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { getSessionsByDate } from '../../services/sessionService';
import logger from '../../utils/logger';
import type { Session } from '../../types/session';

// 페이지 크기 증가
const DEFAULT_PAGE_SIZE = 30;

// 무한 스크롤 트리거 컴포넌트
interface InfiniteScrollTriggerProps {
  isVisible: boolean;
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
  rootElement: HTMLElement | null;
}

function InfiniteScrollTrigger({ 
  isVisible, 
  hasMore, 
  isLoading, 
  onLoadMore, 
  rootElement 
}: InfiniteScrollTriggerProps) {
  const triggerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isVisible || !triggerRef.current || !rootElement) return;
    if (!hasMore || isLoading) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && hasMore && !isLoading) {
          onLoadMore();
        }
      },
      { 
        root: rootElement, 
        rootMargin: '100px', 
        threshold: 0.1 
      }
    );

    observer.observe(triggerRef.current);
    return () => observer.disconnect();
  }, [isVisible, hasMore, isLoading, onLoadMore, rootElement]);

  if (!isVisible || !hasMore) return null;

  return (
    <div ref={triggerRef} className="h-4">
      {isLoading && (
        <div className="flex items-center justify-center py-3 border-t border-slate-100 mt-1">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <LoadingSpinner size="sm" />
            <span>더 불러오는 중...</span>
          </div>
        </div>
      )}
    </div>
  );
}

interface SessionListProps {
  searchQuery?: string;
  onSelect?: (session: Session) => void;
  onRename?: (session: Session) => void;
  onCopy?: (session: Session) => void;
  onExport?: (session: Session) => void;
  onDelete?: (session: Session) => void;
  refreshTrigger?: number;
}

// 날짜 그룹 매핑 함수 (타입 안전)
function getDateGroupParam(group: DateGroup): 'today' | 'yesterday' | 'week' | 'month' | 'older' {
  switch (group) {
    case '오늘':
      return 'today';
    case '어제':
      return 'yesterday';
    case '지난 7일':
      return 'week';
    case '지난 30일':
      return 'month';
    case '이전':
      return 'older';
    default:
      return 'older';
  }
}

// 그룹 데이터 안전 접근 헬퍼 함수
function getGroupData<T>(data: Record<DateGroup, T>, group: DateGroup): T {
  switch (group) {
    case '오늘':
      return data['오늘'];
    case '어제':
      return data['어제'];
    case '지난 7일':
      return data['지난 7일'];
    case '지난 30일':
      return data['지난 30일'];
    case '이전':
      return data['이전'];
    default:
      return data['이전'];
  }
}

// 그룹 데이터 안전 업데이트 헬퍼 함수
function updateGroupData<T>(
  data: Record<DateGroup, T>,
  group: DateGroup,
  updater: (current: T) => T
): Record<DateGroup, T> {
  switch (group) {
    case '오늘':
      return { ...data, '오늘': updater(data['오늘']) };
    case '어제':
      return { ...data, '어제': updater(data['어제']) };
    case '지난 7일':
      return { ...data, '지난 7일': updater(data['지난 7일']) };
    case '지난 30일':
      return { ...data, '지난 30일': updater(data['지난 30일']) };
    case '이전':
      return { ...data, '이전': updater(data['이전']) };
    default:
      return data;
  }
}

// 날짜 그룹 순서
const dateGroupOrder: DateGroup[] = ['오늘', '어제', '지난 7일', '지난 30일', '이전'];

// 그룹별 데이터 타입
interface GroupData {
  sessions: Session[];
  hasMore: boolean;
  page: number;
  isLoading: boolean;
  total: number;
}

export function SessionList({
  searchQuery,
  onSelect,
  onRename,
  onCopy,
  onExport,
  onDelete,
  refreshTrigger,
}: SessionListProps) {
  // 초기에는 '오늘'만 펼치기
  const [expandedGroups, setExpandedGroups] = useState<Set<DateGroup>>(
    new Set(['오늘'])
  );
  
  // 그룹별 데이터 상태
  const [groupData, setGroupData] = useState<Record<DateGroup, GroupData>>({
    '오늘': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
    '어제': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
    '지난 7일': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
    '지난 30일': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
    '이전': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
  });

  // 통합 스크롤 컨테이너 ref
  const mainScrollRef = useRef<HTMLDivElement>(null);

  // 초기 마운트 여부 추적
  const isInitialMount = useRef(true);
  const prevSearchQuery = useRef<string | undefined>(searchQuery);

  // 그룹 세션 로딩 - 페이지 크기 증가
  const loadGroupSessions = useCallback(async (
    group: DateGroup,
    page: number = 1,
    append: boolean = false
  ) => {
    // 함수형 업데이트를 사용하여 현재 상태 확인
    setGroupData(prev => {
      const currentData = getGroupData(prev, group);
      
      // 이미 로딩 중이거나 더 이상 로드할 데이터가 없으면 중단
      if (currentData.isLoading || (!currentData.hasMore && page > 1)) {
        return prev;
      }

      // 로딩 시작
      return updateGroupData(prev, group, (current) => ({
        ...current,
        isLoading: true
      }));
    });

    try {
      const response = await getSessionsByDate(
        getDateGroupParam(group),
        page,
        DEFAULT_PAGE_SIZE,
        searchQuery
      );

      setGroupData(prev => {
        const prevGroupData = getGroupData(prev, group);
        return updateGroupData(prev, group, () => ({
          sessions: append 
            ? [...prevGroupData.sessions, ...response.sessions]
            : response.sessions,
          hasMore: response.sessions.length === DEFAULT_PAGE_SIZE && response.total > prevGroupData.sessions.length + response.sessions.length,
          page,
          isLoading: false,
          total: response.total,
        }));
      });
    } catch (error) {
      logger.error(`Failed to load ${group} sessions:`, error);
      setGroupData(prev => updateGroupData(prev, group, (current) => ({
        ...current,
        isLoading: false
      })));
    }
  }, [searchQuery]);

  // 초기 로딩 및 검색어 변경 처리
  useEffect(() => {
    // 초기 마운트 시: searchQuery가 없을 때 '오늘' 그룹만 로드
    if (isInitialMount.current) {
      isInitialMount.current = false;
      if (!searchQuery) {
        loadGroupSessions('오늘', 1, false);
      }
      prevSearchQuery.current = searchQuery;
      return;
    }

    // 검색어가 실제로 변경된 경우에만 처리
    if (prevSearchQuery.current !== searchQuery) {
      prevSearchQuery.current = searchQuery;
      
      // 모든 그룹의 데이터 초기화
      setGroupData({
        '오늘': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
        '어제': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
        '지난 7일': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
        '지난 30일': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
        '이전': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
      });
      
      // 펼쳐진 그룹들만 다시 로딩 (함수형 업데이트 사용)
      setExpandedGroups(currentExpanded => {
        currentExpanded.forEach(group => {
          loadGroupSessions(group, 1, false);
        });
        return currentExpanded;
      });
    }
  }, [searchQuery, loadGroupSessions]);

  // refreshTrigger 변경 시 세션 목록 새로고침
  useEffect(() => {
    if (refreshTrigger && refreshTrigger > 0) {
      // 펼쳐진 그룹들만 새로고침
      setExpandedGroups(currentExpanded => {
        currentExpanded.forEach(group => {
          loadGroupSessions(group, 1, false);
        });
        return currentExpanded;
      });
    }
  }, [refreshTrigger, loadGroupSessions]);

  // 그룹 토글
  const toggleGroup = useCallback((group: DateGroup) => {
    const isExpanded = expandedGroups.has(group);
    const newExpanded = new Set(expandedGroups);
    
    if (isExpanded) {
      newExpanded.delete(group);
    } else {
      newExpanded.add(group);
      
      // 처음 펼칠 때만 로딩 (함수형 업데이트 사용)
      setGroupData(prev => {
        const currentData = getGroupData(prev, group);
        if (currentData.sessions.length === 0 && !currentData.isLoading) {
          // 비동기 로딩은 useEffect에서 처리
          setTimeout(() => {
            loadGroupSessions(group, 1, false);
          }, 0);
        }
        return prev;
      });
    }
    
    setExpandedGroups(newExpanded);
  }, [expandedGroups, loadGroupSessions]);

  // 통합 스크롤로 다음 그룹 자동 펼치기
  useEffect(() => {
    const container = mainScrollRef.current;
    if (!container || searchQuery) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const scrollBottom = scrollHeight - scrollTop - clientHeight;

      // 하단 300px 전에 다음 그룹 자동 펼치기
      if (scrollBottom < 300) {
        const expandedArray = Array.from(expandedGroups);
        const lastExpandedGroup = expandedArray[expandedArray.length - 1];
        const lastExpandedIndex = lastExpandedGroup 
          ? dateGroupOrder.findIndex(group => group === lastExpandedGroup)
          : -1;

        if (lastExpandedIndex >= 0 && lastExpandedIndex < dateGroupOrder.length - 1) {
          const nextGroup = dateGroupOrder[lastExpandedIndex + 1];
          if (nextGroup && !expandedGroups.has(nextGroup)) {
            setExpandedGroups(prev => {
              const newSet = new Set(prev);
              newSet.add(nextGroup);
              return newSet;
            });
            
            setTimeout(() => {
              loadGroupSessions(nextGroup, 1, false);
            }, 0);
          }
        }
      }
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, [expandedGroups, loadGroupSessions, searchQuery]);

  // 그룹 레이블 생성
  const getGroupLabel = (group: DateGroup): string => {
    const getGroupLabelText = (g: DateGroup): string => {
      switch (g) {
        case '오늘':
          return '오늘';
        case '어제':
          return '어제';
        case '지난 7일':
          return '지난 7일';
        case '지난 30일':
          return '지난 30일';
        case '이전':
          return '이전 대화';
        default:
          return '이전 대화';
      }
    };
    
    const data = getGroupData(groupData, group);
    // total이 있으면 total 사용, 없으면 sessions.length 사용
    // 데이터가 로드되지 않았으면 (total === 0 && sessions.length === 0) 빈 문자열로 표시
    const count = data.total > 0 ? data.total : (data.sessions.length > 0 ? data.sessions.length : 0);
    
    // 데이터가 로드되지 않았고 로딩 중이 아닌 경우 개수 표시 안 함
    if (count === 0 && !data.isLoading && data.sessions.length === 0) {
      return getGroupLabelText(group);
    }
    
    return `${getGroupLabelText(group)} (${count}개)`;
  };

  // 모든 그룹을 항상 표시 (데이터가 없어도 그룹은 표시)
  // 사용자가 그룹을 클릭하면 해당 그룹의 데이터를 로드
  const visibleGroups = dateGroupOrder;

  return (
    <div 
      ref={mainScrollRef}
      className="flex-1 overflow-y-auto"
    >
      <div className="px-4 py-2">
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs font-semibold text-slate-500">대화 히스토리</div>
        </div>

        {visibleGroups.map((group) => {
          const isExpanded = expandedGroups.has(group);
          const data = getGroupData(groupData, group);

          return (
            <div key={group} className="mb-3">
              <button
                onClick={() => toggleGroup(group)}
                className="flex items-center gap-2 w-full text-sm font-medium text-slate-700 hover:bg-slate-100 rounded-lg px-2 py-1.5 transition-colors"
              >
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                <Folder className="w-4 h-4 text-amber-500" />
                {getGroupLabel(group)}
              </button>

              {isExpanded && (
                <div className="mt-1 space-y-1">
                  {data.isLoading && data.sessions.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-8 px-4">
                      <div className="relative">
                        <LoadingSpinner size="md" />
                        <div className="absolute inset-0 flex items-center justify-center">
                          <Inbox className="w-5 h-5 text-blue-400 animate-pulse" />
                        </div>
                      </div>
                      <span className="text-xs text-slate-500 mt-3">대화를 불러오는 중...</span>
                    </div>
                  ) : data.sessions.length === 0 && !data.isLoading ? (
                    <div className="text-xs text-slate-400 text-center py-3 px-2">
                      {searchQuery ? (
                        `"${searchQuery}"에 대한 대화를 찾을 수 없습니다`
                      ) : (
                        '대화 히스토리가 없습니다'
                      )}
                    </div>
                  ) : (
                    <>
                      {data.sessions.map((session) => (
                        <SessionItem
                          key={session.session_id}
                          session={session}
                          onSelect={onSelect}
                          onRename={onRename}
                          onCopy={onCopy}
                          onExport={onExport}
                          onDelete={onDelete}
                        />
                      ))}
                      
                      {/* 무한 스크롤 트리거 */}
                      <InfiniteScrollTrigger
                        isVisible={isExpanded}
                        hasMore={data.hasMore}
                        isLoading={data.isLoading}
                        onLoadMore={() => loadGroupSessions(group, data.page + 1, true)}
                        rootElement={mainScrollRef.current}
                      />
                    </>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* 전체 빈 상태 - 펼쳐진 그룹이 없고 모든 그룹이 비어있을 때만 표시 */}
        {expandedGroups.size === 0 && 
         visibleGroups.length > 0 && 
         !visibleGroups.some(group => {
           const data = getGroupData(groupData, group);
           return data.sessions.length > 0 || data.isLoading;
         }) && 
         !searchQuery && (
          <div className="text-xs text-slate-400 text-center py-8 px-2">
            대화 히스토리가 없습니다
          </div>
        )}
      </div>
    </div>
  );
}

