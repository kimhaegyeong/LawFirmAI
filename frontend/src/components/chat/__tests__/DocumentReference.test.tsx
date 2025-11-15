/**
 * DocumentReference 컴포넌트 테스트
 * disabled prop 및 클릭 제한 테스트
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DocumentReference } from '../DocumentReference';

describe('DocumentReference', () => {
  const mockOnClick = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render document link with correct index', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
      />
    );

    expect(screen.getByText('문서 1')).toBeInTheDocument();
    expect(screen.getByLabelText('문서 1 정보 보기')).toBeInTheDocument();
  });

  it('should call onClick when clicked and not disabled', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
        disabled={false}
      />
    );

    const button = screen.getByRole('button');
    fireEvent.click(button);

    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });

  it('should not call onClick when disabled', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
        disabled={true}
      />
    );

    const button = screen.getByRole('button');
    fireEvent.click(button);

    expect(mockOnClick).not.toHaveBeenCalled();
  });

  it('should have disabled attribute when disabled', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
        disabled={true}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('should have correct title when disabled', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
        disabled={true}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('title', '참고자료 정보를 기다리는 중...');
  });

  it('should have correct title when enabled', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
        disabled={false}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('title', '문서 1 정보 보기');
  });

  it('should apply disabled styles when disabled', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
        disabled={true}
      />
    );

    const button = screen.getByRole('button');
    expect(button.className).toContain('cursor-not-allowed');
    expect(button.className).toContain('bg-slate-100');
  });

  it('should apply enabled styles when not disabled', () => {
    render(
      <DocumentReference
        documentIndex={0}
        onClick={mockOnClick}
        disabled={false}
      />
    );

    const button = screen.getByRole('button');
    expect(button.className).toContain('cursor-pointer');
    expect(button.className).toContain('bg-blue-50');
  });
});

