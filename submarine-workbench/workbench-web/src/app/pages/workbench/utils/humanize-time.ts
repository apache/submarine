export function humanizeTime(time: string){
    //TODO:Translate "unit ago"
    let time_ = time.split(/[\s-:]+/).map(Number);
    let date = new Date(time_[0], time_[1]-1, time_[2], time_[3], time_[4], time_[5]);
    let now = new Date;
    let seconds = (now.getTime() - date.getTime()) / 1000;
    if (seconds <= 0) {
        return 0 + "second ago"
    }
    var numyears = Math.floor(seconds / 31536000);
    if (numyears !== 0) {
        return numyears===1 ? numyears + "year ago" : numyears + "years ago";
    }
    var numdays = Math.floor((seconds % 31536000) / 86400);
    if (numdays !== 0) {
        return numdays===1 ? numdays + "day ago" : numdays + "days ago";
    }
    var numhours = Math.floor(((seconds % 31536000) % 86400) / 3600);
    if (numhours !== 0) {
        return numhours===1 ? numhours + "hour ago" : numhours + "hours ago";
    }
    var numminutes = Math.floor((((seconds % 31536000) % 86400) % 3600) / 60);
    if (numminutes !== 0) {
        return numminutes===1 ? numminutes + "minute ago" : numminutes + "minutes ago";
    }
    var numseconds = (((seconds % 31536000) % 86400) % 3600) % 60;
    return numseconds===1 ? numseconds + "second ago" : numseconds + "seconds ago";
}
