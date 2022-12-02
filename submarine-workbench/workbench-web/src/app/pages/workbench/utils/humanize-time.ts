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
