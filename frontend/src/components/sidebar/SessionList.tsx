/**
 * 세션 리스트 컴포넌트 (하이브리드 로딩)
 */
import { Folder, ChevronDown, ChevronRight } from 'lucide-react';
import { useState, useEffect, useRef, useCallback } from 'react';
import { DateGroup } from '../../utils/dateUtils';
import { SessionItem } from './SessionItem';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { getSessionsByDate } from '../../services/sessionService';
import logger from '../../utils/logger';
import type { Session } from '../../types/session';

interface SessionListProps {
  searchQuery?: string;
  onSelect?: (session: Session) => void;
  onRename?: (session: Session) => void;
  onCopy?: (session: Session) => void;
  onExport?: (session: Session) => void;
  onDelete?: (session: Session) => void;
}

// 날짜 그룹 매핑
const dateGroupMap: Record<DateGroup, 'today' | 'yesterday' | 'week' | 'month' | 'older'> = {
  '오늘': 'today',
  '어제': 'yesterday',
  '지난 7일': 'week',
  '지난 30일': 'month',
  '이전': 'older',
};

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
}: SessionListProps) {
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

  // 그룹별 스크롤 컨테이너 ref
  const groupScrollRefs = useRef<Record<DateGroup, HTMLDivElement | null>>({
    '오늘': null,
    '어제': null,
    '지난 7일': null,
    '지난 30일': null,
    '이전': null,
  });

  // 그룹 세션 로딩
  const loadGroupSessions = useCallback(async (
    group: DateGroup,
    page: number = 1,
    append: boolean = false
  ) => {
    // 함수형 업데이트를 사용하여 현재 상태 확인
    setGroupData(prev => {
      const currentData = prev[group];
      
      // 이미 로딩 중이거나 더 이상 로드할 데이터가 없으면 중단
      if (currentData.isLoading || (!currentData.hasMore && page > 1)) {
        return prev;
      }

      // 로딩 시작
      return {
        ...prev,
        [group]: { ...prev[group], isLoading: true },
      };
    });

    try {
      const response = await getSessionsByDate(
        dateGroupMap[group],
        page,
        20, // page_size
        searchQuery
      );

      setGroupData(prev => ({
        ...prev,
        [group]: {
          sessions: append 
            ? [...prev[group].sessions, ...response.sessions]
            : response.sessions,
          hasMore: response.sessions.length === 20 && response.total > prev[group].sessions.length + response.sessions.length,
          page,
          isLoading: false,
          total: response.total,
        },
      }));
    } catch (error) {
      logger.error(`Failed to load ${group} sessions:`, error);
      setGroupData(prev => ({
        ...prev,
        [group]: { ...prev[group], isLoading: false },
      }));
    }
  }, [searchQuery]);

  // 초기 로딩: 오늘 그룹만
  useEffect(() => {
    loadGroupSessions('오늘', 1, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // 초기 마운트 시에만 실행

  // 검색어 변경 시 모든 그룹 새로고침
  useEffect(() => {
    // 검색어가 변경되면 모든 그룹의 데이터 초기화 및 재로딩
    setGroupData(prev => ({
      '오늘': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
      '어제': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
      '지난 7일': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
      '지난 30일': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
      '이전': { sessions: [], hasMore: true, page: 0, isLoading: false, total: 0 },
    }));
    
    // 펼쳐진 그룹들만 다시 로딩
    // expandedGroups는 클로저로 캡처되므로 현재 상태 사용
    setExpandedGroups(currentExpanded => {
      currentExpanded.forEach(group => {
        setTimeout(() => {
          loadGroupSessions(group, 1, false);
        }, 0);
      });
      return currentExpanded;
    });
  }, [searchQuery, loadGroupSessions]);

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
        const currentData = prev[group];
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

  // 그룹 내 무한 스크롤 처리
  const handleGroupScroll = useCallback((group: DateGroup, element: HTMLDivElement) => {
    const { scrollTop, scrollHeight, clientHeight } = element;
    
    // 함수형 업데이트를 사용하여 현재 상태 확인
    setGroupData(prev => {
      const currentData = prev[group];
      
      // 하단 100px 전에 로딩 시작
      if (
        scrollHeight - scrollTop - clientHeight < 100 &&
        currentData.hasMore &&
        !currentData.isLoading
      ) {
        // 비동기 로딩
        setTimeout(() => {
          loadGroupSessions(group, currentData.page + 1, true);
        }, 0);
      }
      
      return prev;
    });
  }, [loadGroupSessions]);

  // 그룹 레이블 생성
  const getGroupLabel = (group: DateGroup): string => {
    const groupLabels: Record<DateGroup, string> = {
      '오늘': '오늘',
      '어제': '어제',
      '지난 7일': '지난 7일',
      '지난 30일': '지난 30일',
      '이전': '이전 대화',
    };
    
    const data = groupData[group];
    // total이 있으면 total 사용, 없으면 sessions.length 사용
    // 데이터가 로드되지 않았으면 (total === 0 && sessions.length === 0) 빈 문자열로 표시
    const count = data.total > 0 ? data.total : (data.sessions.length > 0 ? data.sessions.length : 0);
    
    // 데이터가 로드되지 않았고 로딩 중이 아닌 경우 개수 표시 안 함
    if (count === 0 && !data.isLoading && data.sessions.length === 0) {
      return groupLabels[group];
    }
    
    return `${groupLabels[group]} (${count}개)`;
  };

  // 모든 그룹을 항상 표시 (데이터가 없어도 그룹은 표시)
  // 사용자가 그룹을 클릭하면 해당 그룹의 데이터를 로드
  const visibleGroups = dateGroupOrder;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="px-4 py-2">
        <div className="text-xs font-semibold text-slate-500 mb-2">대화 히스토리</div>

        {visibleGroups.map((group) => {
          const isExpanded = expandedGroups.has(group);
          const data = groupData[group];

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
                <div
                  ref={(el) => {
                    groupScrollRefs.current[group] = el;
                  }}
                  onScroll={(e) => {
                    if (e.currentTarget) {
                      handleGroupScroll(group, e.currentTarget);
                    }
                  }}
                  className="mt-1 space-y-1 max-h-96 overflow-y-auto"
                >
                  {data.isLoading && data.sessions.length === 0 ? (
                    <div className="flex items-center justify-center py-4">
                      <LoadingSpinner size="sm" />
                      <span className="text-sm text-slate-500 ml-2">로딩 중...</span>
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
                      
                      {data.isLoading && data.sessions.length > 0 && (
                        <div className="flex items-center justify-center py-2">
                          <LoadingSpinner size="sm" />
                          <span className="text-xs text-slate-500 ml-2">더 불러오는 중...</span>
                        </div>
                      )}
                      
                      {!data.hasMore && data.sessions.length > 0 && (
                        <div className="text-xs text-slate-400 text-center py-2">
                          모든 대화를 불러왔습니다
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {visibleGroups.length === 0 && (
          <div className="text-sm text-slate-500 text-center py-8">
            대화 히스토리가 없습니다.
          </div>
        )}
      </div>
    </div>
  );
}

