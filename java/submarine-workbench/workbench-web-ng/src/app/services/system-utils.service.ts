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
import { ListResult, Rest } from '@submarine/interfaces';
import { SysDictItem } from '@submarine/interfaces/sys-dict-item';
import { BaseApiService } from './base-api.service';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';

export enum SysDictCode {
  'USER_SEX' = 'SYS_USER_SEX',
  'USER_STATUS' = 'SYS_USER_STATUS'
}

@Injectable({
  providedIn: 'root'
})
export class SystemUtilsService {
  dictItemCache: { [s: string]: ListResult<any> } = {};

  constructor(private httpClient: HttpClient, private baseApi: BaseApiService) {
  }

  fetchSysDictByCode(code: SysDictCode): Observable<ListResult<SysDictItem>> {
    if (this.dictItemCache[code]) {
      return of(this.dictItemCache[code]);
    }

    const apiUrl = `${this.baseApi.getRestApi('/sys/dictItem/getDictItems')}/${code}`;

    return this.httpClient.get<Rest<ListResult<SysDictItem>>>(apiUrl).pipe(
      switchMap(res => {
        if (res.success) {
          this.dictItemCache[code] = res.result;
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get', code);
        }
      })
    );
  }

  duplicateCheckUsername(userName: string, userId?: string) {
    return this.duplicateCheck('sys_user', 'user_name', userName, userId);
  }

  duplicateCheckUserEmail(email: string, userId?: string) {
    return this.duplicateCheck('sys_user', 'email', email, userId);
  }

  duplicateCheckUserPhone(phone: string, userId?: string) {
    return this.duplicateCheck('sys_user', 'phone', phone, userId);
  }

  private duplicateCheck(
    tableName: string,
    fieldName: string,
    fieldVal: string,
    dataId?: string
  ): Observable<boolean> {
    const apiUrl = this.baseApi.getRestApi('/sys/duplicateCheck');
    const params = {
      tableName,
      fieldName,
      fieldVal,
      dataId: dataId
    };

    return this.httpClient.get<Rest<string>>(apiUrl, {
      params
    }).pipe(
      switchMap(res => {
        return of(res.success);
      })
    );
  }
}
