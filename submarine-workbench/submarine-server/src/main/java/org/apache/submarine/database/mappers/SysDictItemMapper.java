/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.database.mappers;

import org.apache.ibatis.session.RowBounds;
import org.apache.submarine.database.entity.SysDictItem;

import java.util.List;
import java.util.Map;

public interface SysDictItemMapper {
  List<SysDictItem> selectAll(Map where, RowBounds rowBounds);

  int insertSysDictItem(SysDictItem sysDictItem);

  boolean updateBy(SysDictItem sysDictItem);

  SysDictItem getById(String id);

  void deleteById(String id);

  List<SysDictItem> queryDictByCode(String dictCode);
}
