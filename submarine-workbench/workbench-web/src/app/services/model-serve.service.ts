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
import { Observable, of, Subject, throwError } from 'rxjs';
import { BaseApiService } from '@submarine/services/base-api.service';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ServeSpec } from '@submarine/interfaces/model-serve';
import { Rest } from '@submarine/interfaces';
import { catchError, map, switchMap } from 'rxjs/operators';

@Injectable({
    providedIn: 'root',
})

export class ModelServeService {
    private emitInfoSource = new Subject<string>();
  infoEmitted$ = this.emitInfoSource.asObservable();

  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  emitInfo(id: string) {
    this.emitInfoSource.next(id);
  }

  createServe(modelName: string, modelVersion: number) : Observable<string> {
    const apiUrl = this.baseApi.getRestApi('/v1/serve');
    const serveSpec : ServeSpec = {
      modelName,
      modelVersion
    }
    return this.httpClient.post<Rest<any>>(apiUrl, serveSpec).pipe(
      map((res) => res.result),
      catchError((e) => {
        console.log(e);
        let message: string;
        if (e.error instanceof ErrorEvent) {
          // client side error
          message = 'Something went wrong with network or workbench';
        } else {
          if (e.status >= 500) {
            message = `${e.message}`;
          } else {
            message = e.error.message;
          }
        }
        return throwError(message);
      })
    );
  }

  deleteServe(modelName: string, modelVersion: number) : Observable<string> {
    const apiUrl = this.baseApi.getRestApi(`/v1/serve`);
    const serveSpec : ServeSpec = {
      modelName,
      modelVersion
    }
    const options = {
      headers: new HttpHeaders({
        'Content-Type': 'application/json',
      }),
      body: serveSpec,
    };
    return this.httpClient.delete<Rest<any>>(apiUrl, options).pipe(
      map((res) => res.result),
      catchError((e) => {
        console.log(e);
        let message: string;
        if (e.error instanceof ErrorEvent) {
          // client side error
          message = 'Something went wrong with network or workbench';
        } else {
          if (e.status >= 500) {
            message = `${e.message}`;
          } else {
            message = e.error.message;
          }
        }
        return throwError(message);
      })
    )
  }

}