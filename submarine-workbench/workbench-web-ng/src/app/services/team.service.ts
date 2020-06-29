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
import { ListResult, Rest } from '@submarine/interfaces';
import { SysTeam } from '@submarine/interfaces/sys-team';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { BaseApiService } from './base-api.service';

interface TeamListQueryParams {
  teamName: string;
  owner: string;
  column: string;
  order: string;
}

@Injectable({
  providedIn: 'root'
})
export class TeamService {
  constructor(private httpClient: HttpClient, private baseApi: BaseApiService) {}

  getTeamList(queryParams: Partial<TeamListQueryParams>): Observable<ListResult<SysTeam>> {
    const apiUrl = this.baseApi.getRestApi('/team/list');
    return this.httpClient
      .get<Rest<ListResult<SysTeam>>>(apiUrl, {
        params: queryParams
      })
      .pipe(
        switchMap((res) => {
          if (res.success) {
            console.log(res.result);
            return of(res.result);
          } else {
            throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get', queryParams);
          }
        })
      );
  }

  createTeam(params): Observable<SysTeam> {
    const apiUrl = this.baseApi.getRestApi('/team/add');
    return this.httpClient.post<Rest<SysTeam>>(apiUrl, params).pipe(
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

  deleteTeam(id: string) {
    const apiUrl = this.baseApi.getRestApi('/team/delete');
    return this.httpClient
      .delete<Rest<any>>(apiUrl, {
        params: {
          id
        }
      })
      .pipe(
        switchMap((res) => {
          if (res.success) {
            return of(true);
          } else {
            throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'post', id);
          }
        })
      );
  }

  newTeamNameCheck(nameParams): Promise<ValidationErrors | null> {
    const promise = new Promise((resolve, reject) => {
      const apiUrl = this.baseApi.getRestApi('/sys/duplicateCheck');
      this.httpClient
        .get<any>(apiUrl, {
          params: nameParams
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
}
