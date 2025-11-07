/**
 * 파일 첨부 컴포넌트
 */
import { X, FileText, FileImage } from 'lucide-react';
import { formatFileSize, isImageFile } from '../../utils/fileUtils';
import type { FileAttachment as FileAttachmentType } from '../../types/chat';

interface FileAttachmentProps {
  attachment: FileAttachmentType;
  onRemove?: (id: string) => void;
  showRemove?: boolean;
}

export function FileAttachment({
  attachment,
  onRemove,
  showRemove = true,
}: FileAttachmentProps) {
  return (
    <div className="relative group">
      <div className="flex items-center gap-2 bg-blue-50 border border-blue-200 pl-3 pr-8 py-2 rounded-lg">
        {attachment.url && isImageFile(attachment.type) ? (
          <img
            src={attachment.url}
            alt={attachment.name}
            className="w-10 h-10 object-cover rounded"
          />
        ) : (
          <div className="w-10 h-10 bg-blue-100 rounded flex items-center justify-center">
            <FileText className="w-5 h-5 text-blue-600" />
          </div>
        )}
        <div className="flex flex-col">
          <span className="text-sm font-medium text-slate-700 max-w-[200px] truncate">
            {attachment.name}
          </span>
          <span className="text-xs text-slate-500">
            {formatFileSize(attachment.size)}
          </span>
        </div>
        {showRemove && onRemove && (
          <button
            onClick={() => onRemove(attachment.id)}
            className="absolute right-1 top-1/2 -translate-y-1/2 p-1 bg-white rounded-full hover:bg-red-50 shadow-sm transition-colors"
          >
            <X className="w-3.5 h-3.5 text-slate-600 hover:text-red-600" />
          </button>
        )}
      </div>
    </div>
  );
}

