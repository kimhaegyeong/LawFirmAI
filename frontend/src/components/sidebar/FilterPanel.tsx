/**
 * 필터 패널 컴포넌트
 */
import { SESSION_CATEGORIES, SORT_OPTIONS } from '../../utils/constants';
import type { SessionCategory, SortBy, SortOrder } from '../../types/session';

interface FilterPanelProps {
  category?: SessionCategory;
  sortBy?: SortBy;
  sortOrder?: SortOrder;
  onCategoryChange?: (category: SessionCategory) => void;
  onSortChange?: (sortBy: SortBy, sortOrder: SortOrder) => void;
}

export function FilterPanel({
  category = '전체',
  sortBy = 'updated_at',
  sortOrder = 'desc',
  onCategoryChange,
  onSortChange,
}: FilterPanelProps) {
  return (
    <div className="p-4 border-t border-slate-200">
      <div className="mb-4">
        <div className="text-xs font-semibold text-slate-500 mb-2">카테고리</div>
        <div className="space-y-1">
          {SESSION_CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => onCategoryChange?.(cat)}
              className={`w-full text-left px-2 py-1.5 text-sm rounded-lg transition-colors ${
                category === cat
                  ? 'bg-blue-100 text-blue-700 font-medium'
                  : 'text-slate-700 hover:bg-slate-100'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="text-xs font-semibold text-slate-500 mb-2">정렬</div>
        <div className="space-y-1">
          {SORT_OPTIONS.map((option) => (
            <button
              key={option.value}
              onClick={() =>
                onSortChange?.(
                  option.value as SortBy,
                  sortBy === option.value && sortOrder === 'desc' ? 'asc' : 'desc'
                )
              }
              className={`w-full text-left px-2 py-1.5 text-sm rounded-lg transition-colors ${
                sortBy === option.value
                  ? 'bg-blue-100 text-blue-700 font-medium'
                  : 'text-slate-700 hover:bg-slate-100'
              }`}
            >
              {option.label}
              {sortBy === option.value && (
                <span className="ml-1 text-xs">{sortOrder === 'desc' ? '↓' : '↑'}</span>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

