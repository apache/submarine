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
package org.apache.submarine.server.utils.response;

import net.sf.cglib.beans.BeanGenerator;
import net.sf.cglib.beans.BeanMap;
import org.apache.commons.lang.StringUtils;
import org.apache.submarine.server.utils.response.JsonResponse.ListResult;
import org.apache.submarine.server.database.workbench.annotation.Dict;
import org.apache.submarine.server.database.workbench.entity.SysDictItemEntity;
import org.apache.submarine.server.database.workbench.service.SysDictItemService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

// Dict Annotation
public class DictAnnotation {
  private static final Logger LOG = LoggerFactory.getLogger(DictAnnotation.class);

  public static final String DICT_SUFFIX = "@dict";

  // Dynamically generated class
  private Object object = null;

  // Attribute name and type of attribute
  private BeanMap beanMap = null;

  public DictAnnotation(Map propertyMap) {
    this.object = generateBean(propertyMap);
    this.beanMap = BeanMap.create(this.object);
  }

  private Object generateBean(Map mapProperty) {
    BeanGenerator generator = new BeanGenerator();

    Set keySet = mapProperty.keySet();
    for (Object elem : keySet) {
      String key = (String) elem;
      generator.addProperty(key, (Class) mapProperty.get(key));
    }
    return generator.create();
  }

  public void setValue(Object property, Object value) {
    beanMap.put(property, value);
  }

  public Object getObject() {
    return this.object;
  }

  public static Field[] getAllFields(Object object) {
    Class<?> clazz = object.getClass();
    List<Field> fieldList = new ArrayList<>();
    while (clazz != null) {
      fieldList.addAll(new ArrayList<>(Arrays.asList(clazz.getDeclaredFields())));
      clazz = clazz.getSuperclass();
    }
    Field[] fields = new Field[fieldList.size()];
    fieldList.toArray(fields);
    return fields;
  }

  private static Object mergeDictText(Object object, Map<String, List<SysDictItemEntity>> mapDictItems)
      throws Exception {
    // Map<Field->Value>
    HashMap<String, Object> mapFieldValues = new HashMap<>();
    //  Map<Field->FieldType>
    HashMap<String, Object> mapFieldAndType = new HashMap<>();

    Class<? extends Object> objectClass = object.getClass();
    BeanInfo beanInfo = Introspector.getBeanInfo(objectClass);
    PropertyDescriptor[] propertyDescriptors = beanInfo.getPropertyDescriptors();
    // Get data that already exists in the object
    for (PropertyDescriptor descriptor : propertyDescriptors) {
      String propertyName = descriptor.getName();
      if (!propertyName.equals("class")) {
        Method readMethod = descriptor.getReadMethod();
        if (null == readMethod) {
          throw new Exception("Can not found " + propertyName + " ReadMethod(), All fields in "
              + objectClass.getName() + " need add set and set methods.");
        }
        Object result = readMethod.invoke(object, new Object[0]);
        mapFieldValues.put(propertyName, result);
        mapFieldAndType.put(propertyName, descriptor.getPropertyType());

        if (mapDictItems.containsKey(propertyName)) {
          // add new dict text field to object
          mapFieldAndType.put(propertyName + DICT_SUFFIX, String.class);

          List<SysDictItemEntity> dictItems = mapDictItems.get(propertyName);
          for (SysDictItemEntity dictItem : dictItems) {
            if (StringUtils.equals(String.valueOf(result), dictItem.getItemCode())) {
              mapFieldValues.put(propertyName + DICT_SUFFIX, dictItem.getItemName());
              break;
            }
          }
        }
      }
    }

    // Map to entity object
    DictAnnotation bean = new DictAnnotation(mapFieldAndType);
    Set<String> keys = mapFieldAndType.keySet();
    for (String key : keys) {
      bean.setValue(key, mapFieldValues.get(key));
    }

    return bean.getObject();
  }

  public static boolean parseDictAnnotation(Object result) throws Exception {
    List<Object> dicts = new ArrayList<>();

    if (result instanceof ListResult) {
      ListResult listResult = (ListResult) result;
      if (listResult.getTotal() == 0) {
        return false;
      }

      // Query all the dictionaries that need to be parsed at once
      Map<String, List<SysDictItemEntity>> mapDictItems = new HashMap<>();
      SysDictItemService sysDictItemService = new SysDictItemService();
      Object object = listResult.getRecords().get(0);
      Map<String, Map<String, String>> dictLib = new HashMap<>();
      for (Field field : getAllFields(object)) {
        if (field.getAnnotation(Dict.class) != null) {
          String code = field.getAnnotation(Dict.class).Code();
          List<SysDictItemEntity> dictItems = sysDictItemService.queryDictByCode(code);
          if (dictItems.size() > 0) {
            mapDictItems.put(field.getName(), dictItems);
          }
        }
      }

      if (mapDictItems.size() == 0) {
        // don't contain dict Annotation
        return false;
      }

      for (Object record : listResult.getRecords()) {
        Object newObj = mergeDictText(record, mapDictItems);
        dicts.add(newObj);
      }
      listResult.setRecords(dicts);

      return true;
    } else {
      // When it contains lists, mostly arraylists and the like,
      // we should do as a trace/debug level as it does not affect the data returned
      LOG.trace("Unsupported parse {} Dict Annotation!", result.getClass());
    }

    return false;
  }
}
