/**
 * 인증 관련 타입 정의
 */

export interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  token_type: string;
  expires_in: number;
}

export interface UserInfo {
  user_id: string;
  email?: string;
  name?: string;
  picture?: string;
  provider?: string;
  authenticated: boolean;
}

export interface AuthState {
  user: UserInfo | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: Error | null;
}

export interface DeleteAccountResponse {
  message: string;
  deleted_sessions: number;
}

