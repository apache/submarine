/*!
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

import { Component, OnInit } from '@angular/core';
import { NzMessageService } from 'ng-zorro-antd/message';

@Component({
  selector: 'submarine-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  // TODO(kevin85421): mock data for status
  openRecentPageIndex: number;
  numRunningSession: number;
  numRunningJob: number;
  usedGPUpercent: number;
  usedCPUpercent: number;
  usedMemory: number;
  totalMemory: number;
  usedMemoryPercent: number;

  // TODO(kevin85421): mock data for news
  publishedTime: number = Date.now();
  newsPageSize: number;
  newsPageIndex: number;
  displayNewsList = [];
  newsList = [
    {
      title: 'Submarine supports yarn 2.7.x',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 123',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 123',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 123',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 123',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 123',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 456',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 456',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 456',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 456',
      newsTime:  this.publishedTime,
      iconType: 'user'
    },
    {
      title: 'Submarine supports yarn 2.7.x 456',
      newsTime:  this.publishedTime,
      iconType: 'user'
    }
  ];

  // TODO(kevin85421): mock data for open recent
  openRecentList = [
    {
      title: 'Project1',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project1',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project1',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project1',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project1',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project1',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project2',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project2',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project2',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project2',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project2',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project2',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project3',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project3',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project3',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project3',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project3',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    },
    {
      title: 'Project3',
      description:  'This is the description',
      avatarType: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png'
    }
  ];

  displayOpenRecentList = []
  openRecentPageSize: number;

  // constructor
  constructor(private nzMessageService: NzMessageService) {
    // open recent parameters
    this.openRecentPageIndex = 1;
    this.openRecentPageSize = 6;
    const startOpenRecentIndex = (this.openRecentPageIndex - 1) * this.openRecentPageSize;
    const endOpenRecentIndex = this.openRecentPageIndex * this.openRecentPageSize;
    for (let i = startOpenRecentIndex; i < this.openRecentList.length; i++) {
      if (i >= startOpenRecentIndex && i < endOpenRecentIndex) {
        this.displayOpenRecentList.push(this.openRecentList[i]);
      } else {
        break;
      }
    }

    // status parameters
    this.numRunningSession = 10;
    this.numRunningJob = 6;
    this.usedGPUpercent = 35;
    this.usedCPUpercent = 72;
    this.usedMemory = 12;
    this.totalMemory = 17;
    this.usedMemoryPercent = (this.usedMemory * 100) / this.totalMemory;

    // news parameters
    this.newsPageIndex = 1;
    this.newsPageSize = 5;
    const startNewsIndex = (this.newsPageIndex - 1) * this.newsPageSize;
    const endNewsIndex = this.newsPageIndex * this.newsPageSize;
    for (let i = startNewsIndex; i < this.newsList.length; i++) {
      if (i >= startNewsIndex && i < endNewsIndex) {
        this.displayNewsList.push(this.newsList[i]);
      } else {
        break;
      }
    }
  }

  ngOnInit() {
  }

  openRecentChangePage(event: number) {
    const pageIndex: number = event;
    const startOpenRecentIndex = (pageIndex - 1) * this.openRecentPageSize;
    const endOpenRecentIndex = pageIndex * this.openRecentPageSize;
    this.displayOpenRecentList = [];
    for (let i = startOpenRecentIndex; i < this.openRecentList.length; i++) {
      if (i >= startOpenRecentIndex && i < endOpenRecentIndex) {
        this.displayOpenRecentList.push(this.openRecentList[i]);
      } else {
        break;
      }
    }
  }

  newsChangePage(event: number) {
    const pageIndex: number = event;
    const startNewsIndex = (pageIndex - 1) * this.newsPageSize;
    const endNewsIndex = pageIndex * this.newsPageSize;
    this.displayNewsList = [];
    for (let i = startNewsIndex; i < this.newsList.length; i++) {
      if (i >= startNewsIndex && i < endNewsIndex) {
        this.displayNewsList.push(this.newsList[i]);
      } else {
        break;
      }
    }
  }
}
