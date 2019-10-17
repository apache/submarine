import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Rest, User } from '@submarine/interfaces';
import { BaseApiService } from '@submarine/services/base-api.service';
import { LocalStorageService } from '@submarine/services/local-storage.service';
import * as md5 from 'md5';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  isLoggedIn = false;
  authTokenKey = 'auth_token';

  // store the URL so we can redirect after logging in
  redirectUrl: string;

  constructor(
    private localStorageService: LocalStorageService,
    private baseApi: BaseApiService,
    private httpClient: HttpClient
  ) {
    const authToken = this.localStorageService.get<string>(this.authTokenKey);
    this.isLoggedIn = !!authToken;
  }

  login(userForm: { userName: string; password: string }): Observable<boolean> {
    return this.httpClient
      .post<Rest<User>>(this.baseApi.getRestApi('/auth/login'), {
        username: userForm.userName,
        password: md5(userForm.password)
      })
      .pipe(
        map(res => {
          if (res.success) {
            this.isLoggedIn = true;
            this.localStorageService.set(this.authTokenKey, res.result.token);
          }

          return res.success;
        })
      );
  }

  logout(): void {
    this.isLoggedIn = false;
    this.localStorageService.remove(this.authTokenKey);
  }
}
