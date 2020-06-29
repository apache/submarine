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
import { ValidationErrors } from '@angular/forms';
import { Rest } from '@submarine/interfaces';
import { SysDeptItem } from '@submarine/interfaces/sys-dept-item';
import { SysDeptSelect } from '@submarine/interfaces/sys-dept-select';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { BaseApiService } from './base-api.service';

@Injectable({
  providedIn: 'root'
})
export class DepartmentService {
  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  fetchSysDeptSelect(): Observable<SysDeptSelect[]> {
    const apiUrl = this.baseApi.getRestApi('/sys/dept/queryIdTree');
    return this.httpClient.get<Rest<SysDeptSelect[]>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  fetchSysDeptList(): Observable<SysDeptItem[]> {
    const apiUrl = this.baseApi.getRestApi('/sys/dept/tree');
    return this.httpClient.get<Rest<any>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          console.log(res.result);
          return of(res.result.records);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  codeCheck(codeParams): Promise<ValidationErrors | null> {
    const promise = new Promise((resolve, reject) => {
      const apiUrl = this.baseApi.getRestApi('/sys/duplicateCheck');
      this.httpClient
        .get<any>(apiUrl, {
          params: codeParams
        })
        .toPromise()
        .then(
          (res: any) => {
            console.log(res);
            resolve(res.success);
          },
          (err) => {
            console.log(err);
            reject(err);
          }
        );
    });
    return promise;
  }

  createDept(params): Observable<SysDeptItem> {
    const apiUrl = this.baseApi.getRestApi('/sys/dept/add');
    return this.httpClient.post<Rest<SysDeptItem>>(apiUrl, params).pipe(
      switchMap((res) => {
        console.log(res);
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'post', params);
        }
      })
    );
  }

  editDept(params): Observable<boolean> {
    const apiUrl = this.baseApi.getRestApi('/sys/dept/edit');
    return this.httpClient.put<Rest<any>>(apiUrl, params).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(true);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'put', params);
        }
      })
    );
  }
}
