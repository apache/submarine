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

import org.apache.submarine.database.entity.SysUser;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static junit.framework.TestCase.assertNotNull;

public class SysUserMapperTest {
  private static final Logger LOG = LoggerFactory.getLogger(SysUserMapperTest.class);

  private SysUserMapper sysUserMapper = new SysUserMapper();

  @Test
  public void getUserByNameTest() {
    String username = "admin";
    String password = "21232f297a57a5a743894a0e4a801fc3";

    SysUser sysUser = sysUserMapper.getUserByName(username, password);
    assertNotNull("Cannot get data from the database", sysUser);
    LOG.info(sysUser.toString());
  }
}
