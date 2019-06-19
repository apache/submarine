/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package org.apache.submarine.tony.models;

import org.apache.submarine.tony.events.avro.Event;
import org.apache.submarine.tony.events.avro.EventType;
import java.util.Date;


public class JobEvent {
  private EventType type;
  private Object event;
  private long timestamp;

  public EventType getType() {
    return type;
  }

  public Object getEvent() {
    return event;
  }

  public Date getTimestamp() {
    return new Date(timestamp);
  }

  public void setType(EventType type) {
    this.type = type;
  }

  public void setEvent(Object event) {
    this.event = event;
  }

  public void setTimestamp(long timestamp) {
    this.timestamp = timestamp;
  }

  public static JobEvent convertEventToJobEvent(Event e) {
    JobEvent wrapper = new JobEvent();
    wrapper.setType(e.getType());
    wrapper.setEvent(e.getEvent());
    wrapper.setTimestamp(e.getTimestamp());
    return wrapper;
  }
}
