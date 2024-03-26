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

import {TranslateService} from "@ngx-translate/core";

export function humanizeTime(time: string, translate: TranslateService) {
    let time_ = time.split(/[\s-:]+/).map(Number);
    let date = new Date(time_[0], time_[1] - 1, time_[2], time_[3], time_[4], time_[5]);
    let now = new Date;
    let seconds = (now.getTime() - date.getTime()) / 1000;
    if (seconds <= 0) {
      return 0 + translate.instant("second ago")
    }
    const numyears = Math.floor(seconds / 31536000);
    if (numyears !== 0) {
      return numyears + translate.instant(numyears === 1 ? "year ago" : "years ago");
    }
    const numdays = Math.floor((seconds % 31536000) / 86400);
    if (numdays !== 0) {
      return numdays + translate.instant(numdays === 1 ? "day ago" : "days ago");
    }
    const numhours = Math.floor(((seconds % 31536000) % 86400) / 3600);
    if (numhours !== 0) {
      return numhours + translate.instant(numhours === 1 ? "hour ago" : "hours ago");
    }
    const numminutes = Math.floor((((seconds % 31536000) % 86400) % 3600) / 60);
    if (numminutes !== 0) {
      return numminutes + translate.instant(numminutes === 1 ? "minute ago" : "minutes ago");
    }
    const numseconds = (((seconds % 31536000) % 86400) % 3600) % 60;
    return numseconds + translate.instant(numseconds === 1 ? "second ago" : "seconds ago");
}
