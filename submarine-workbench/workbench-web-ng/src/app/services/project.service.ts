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
import { ListResult, Rest, Project } from '@submarine/interfaces';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { BaseApiService } from './base-api.service';

interface ProjectQueryParams {
  userName: string;
  column: string;
  order: string;
  pageNo: string;
  pageSize: string;
}
interface AddProjectParams {
  name: string;
  userName: string;
  description: string;
  type: string;
  teamName: string;
  visibility: string;
  permission: string;
  starNum: number;
  likeNum: number;
  messageNum: number;
}

@Injectable({
  providedIn: 'root'
})
export class ProjectService {

  constructor(
    private baseApi: BaseApiService,
    private httpClient: HttpClient
  ) {

  }

  fetchProjectList(queryParams: Partial<ProjectQueryParams>): Observable<ListResult<Project>> {
    const apiUrl = this.baseApi.getRestApi('/project/list');
    console.log(apiUrl)
    return this.httpClient.get<Rest<ListResult<Project>>>(apiUrl, {params: queryParams})
    .pipe(
      switchMap(res => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get', queryParams);
        }
      })
    );;

  }

  addProject(params: Partial<AddProjectParams>): Observable<Project> {
    console.log("addProject", params)
    const apiUrl = this.baseApi.getRestApi('/project/add');
    return this.httpClient.post<Rest<Project>>(apiUrl, params)
    .pipe(
      switchMap(res => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'post', params);
        }
      })
    );
  }
}
