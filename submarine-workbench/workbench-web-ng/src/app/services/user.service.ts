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

import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { ListResult, Rest, SysUser, UserInfo } from '@submarine/interfaces';
import * as md5 from 'md5';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { BaseApiService } from './base-api.service';

interface UserListQueryParams {
  accountName: string;
  email: string;
  deptCode: string;
  column: string;
  order: string;
  field: string;
  pageNo: string;
  pageSize: string;
}

@Injectable({
  providedIn: 'root'
})
export class UserService {
  private userInfo: UserInfo;

  constructor(private httpClient: HttpClient, private baseApi: BaseApiService) {
  }

  fetchUserInfo(): Observable<UserInfo> {
    const apiUrl = this.baseApi.getRestApi('/sys/user/info');
    return this.httpClient.get<Rest<UserInfo>>(apiUrl).pipe(
      switchMap(res => {
        if (res.success) {
          this.userInfo = new UserInfo(res.result);
          return of(this.userInfo);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  fetchUserList(queryParams: Partial<UserListQueryParams>): Observable<ListResult<SysUser>> {
    const apiUrl = this.baseApi.getRestApi('/sys/user/list');
    return this.httpClient.get<Rest<ListResult<SysUser>>>(apiUrl, {
      params: queryParams
    }).pipe(
      switchMap(res => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get', queryParams);
        }
      })
    );
  }

  changePassword(id: string, password: string): Observable<boolean> {
    const apiUrl = this.baseApi.getRestApi('/sys/user/changePassword');

    return this.httpClient.put<Rest<any>>(apiUrl, {
      id,
      password: md5(password)
    }).pipe(
      switchMap(res => {
        if (res.success) {
          return of(true);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'put', { id, password });
        }
      })
    );
  }

  createUser(sysUser: Partial<SysUser>): Observable<SysUser> {
    const apiUrl = this.baseApi.getRestApi('/sys/user/add');

    return this.httpClient.post<Rest<SysUser>>(apiUrl, sysUser).pipe(
      switchMap(res => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'post', sysUser);
        }
      })
    );
  }

  updateUser(sysUser: Partial<SysUser>): Observable<SysUser> {
    const apiUrl = this.baseApi.getRestApi('/sys/user/edit');

    return this.httpClient.put<Rest<SysUser>>(apiUrl, sysUser).pipe(
      switchMap(res => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'put', sysUser);
        }
      })
    );
  }

  deleteUser(id: string): Observable<boolean> {
    const apiUrl = this.baseApi.getRestApi(`/sys/user/delete`);

    return this.httpClient.delete<Rest<any>>(apiUrl, {
      params: {
        id
      }
    }).pipe(
      switchMap(res => {
        if (res.success) {
          return of(true);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'delete', id);
        }
      })
    );
  }
}
