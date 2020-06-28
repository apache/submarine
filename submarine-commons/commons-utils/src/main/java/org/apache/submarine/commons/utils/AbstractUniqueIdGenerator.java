package org.apache.submarine.commons.utils;

public abstract class AbstractUniqueIdGenerator<T> implements Comparable<T> {
  
  private static final int ID_MIN_DIGITS = 4;
  
  private int id;

  private long serverTimestamp;  
  
  /**
   * Get the count since the server started
   * @return number
   */
  public int getId() {
    return id;
  }

  /**
   * Set the count of job
   * @param id number
   */
  public void setId(int id) {
    this.id = id;
  }

  /**
   * Get the timestamp(s) when the server started
   * @return timestamp(s)
   */
  public long getServerTimestamp() {
    return serverTimestamp;
  }

  /**
   * Set the server started timestamp(s)
   * @param timestamp seconds
   */
  public void setServerTimestamp(long timestamp) {
    this.serverTimestamp = timestamp;
  }
  
  @SuppressWarnings("rawtypes")
  @Override
  public int compareTo(Object o) {
    AbstractUniqueIdGenerator other = (AbstractUniqueIdGenerator) o;
    return this.getId() > other.getId() ? 1 : 0;
  }
  
  @Override
  public int hashCode() {
    final int prime = 371237;
    int result = 6521;
    result = prime * result + (int) (getServerTimestamp() ^ (getServerTimestamp() >>> 32));
    result = prime * result + getId();
    return result;
  }

  @SuppressWarnings("rawtypes")
  @Override
  public boolean equals(Object obj) {
    if (obj == null || getClass() != obj.getClass()) {
      return false;
    }
    if (this == obj) {
      return true;
    }
    AbstractUniqueIdGenerator other = (AbstractUniqueIdGenerator) obj;
    if (this.getServerTimestamp() != other.getServerTimestamp()) {
      return false;
    }
    return this.getId() == other.getId();
  }


  protected void format(StringBuilder sb, long value) {
    int minimumDigits = ID_MIN_DIGITS;
    if (value < 0) {
      sb.append('-');
      value = -value;
    }

    long tmp = value;
    do {
      tmp /= 10;
    } while (--minimumDigits > 0 && tmp > 0);

    for (int i = minimumDigits; i > 0; --i) {
      sb.append('0');
    }
    sb.append(value);
  }
}
