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
import { BaseApiService } from './base-api.service';
import { ListResult, Rest } from '@submarine/interfaces';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { SysTeam } from '@submarine/interfaces/sys-team';

interface TeamListQueryParams {
  teamName: string;
  owner: string;
  column: string;
  order: string;
  pageNo: string;
  pageSize: string;
}

@Injectable({
  providedIn: 'root'
})

export class TeamService {

  constructor(private httpClient: HttpClient, private baseApi: BaseApiService) {
  }

  getTeamList(queryParams: Partial<TeamListQueryParams>): Observable<ListResult<SysTeam>> {
    const apiUrl = this.baseApi.getRestApi('/team/list');
    return this.httpClient.get<Rest<ListResult<SysTeam>>>(apiUrl, {
      params: queryParams
    }).pipe(
      switchMap(res => {
        if (res.success) {
          console.log(res.result);
          return of(res.result);
        }
        else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get', queryParams);
        }
      })
    );
  }
  
}
