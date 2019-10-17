export interface Rest<T> {
  result: T;
  code: number;
  message: string;
  status: string;
  success: boolean;
}
