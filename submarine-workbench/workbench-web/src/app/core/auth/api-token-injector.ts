/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import {HttpErrorResponse, HttpEvent, HttpHandler, HttpInterceptor, HttpRequest} from '@angular/common/http';
import {Injectable} from '@angular/core';
import {Router} from '@angular/router';
import {of, throwError, Observable} from 'rxjs';
import {catchError} from "rxjs/operators";
import {AuthService} from "../../services";

@Injectable()
export class ApiTokenInjector implements HttpInterceptor {

  constructor(private authService: AuthService, private router: Router) { }

  private handleAuthError(err: HttpErrorResponse): Observable<any> {
    // handle auth error or rethrow
    if (err.status === 401 || err.status === 403) {
      if ("session" !== this.authService.getFlowType()) {
        // remove token cache
        this.authService.removeToken();
        // navigate to login
        this.router.navigate(['/user/login']);
      }
      return of(err.message);
    }
    return throwError(err);
  }

  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    // If there is a token in localstorage and not with session, set the token into the header
    const checkToken = "session" !== this.authService.getFlowType() && !!this.authService.getToken();
    let handler;
    if (checkToken) {
      handler = next.handle(request.clone({
        setHeaders: {Authorization: `Bearer ${this.authService.getToken()}`}
      }));
    } else {
      handler = next.handle(request);
    }
    // handle unauthorized exception (like 401)
    return handler.pipe(catchError(x => this.handleAuthError(x)));
  }
}
