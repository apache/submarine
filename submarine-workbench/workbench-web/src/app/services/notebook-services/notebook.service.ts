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

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Rest } from '@submarine/interfaces';
import { BaseApiService } from '@submarine/services/base-api.service';
import { of, throwError, Observable } from 'rxjs';
import { catchError, map, switchMap } from 'rxjs/operators';
import { NotebookInfo } from '@submarine/interfaces/notebook-interfaces/notebook-info';
import { NotebookSpec } from '@submarine/interfaces/notebook-interfaces/notebook-spec';

@Injectable({
  providedIn: 'root',
})
export class NotebookService {
  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  fetchNotebookList(id: string): Observable<NotebookInfo[]> {
    const apiUrl = this.baseApi.getRestApi('/v1/notebook?id=' + id);
    return this.httpClient.get<Rest<NotebookInfo[]>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  createNotebook(notebookSpec: object): Observable<NotebookSpec> {
    const apiUrl = this.baseApi.getRestApi('/v1/notebook');
    return this.httpClient.post<Rest<NotebookSpec>>(apiUrl, notebookSpec).pipe(
      map((res) => res.result), // return result directly if succeeding
      catchError((e) => {
        let message: string;
        if (e.error instanceof ErrorEvent) {
          // client side error
          message = 'Something went wrong with network or workbench';
        } else {
          console.log(e);
          if (e.status === 409) {
            message = 'You might have a duplicate notebook name';
          } else if (e.status >= 500) {
            message = `${e.message}`;
          } else {
            message = e.error.message;
          }
        }
        return throwError(message);
      })
    );
  }

  deleteNotebook(id: string): Observable<NotebookInfo> {
    const apiUrl = this.baseApi.getRestApi(`/v1/notebook/${id}`);
    return this.httpClient.delete<Rest<NotebookInfo>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'delete', id);
        }
      })
    );
  }
}
