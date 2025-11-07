/**
 * 사이드바 컴포넌트
 */
import { ReactNode } from 'react';

interface SidebarProps {
  isOpen: boolean;
  children: ReactNode;
}

export function Sidebar({ isOpen, children }: SidebarProps) {
  return (
    <div
      className={`${
        isOpen ? 'w-64' : 'w-0'
      } bg-white border-r border-slate-200 transition-all duration-300 overflow-hidden flex flex-col`}
    >
      {children}
    </div>
  );
}

