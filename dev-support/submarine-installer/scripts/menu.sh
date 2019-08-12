#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## @description  main menu
## @audience     public
## @stability    stable
main_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}            DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.prepare system environment [..]\\e[0m"
echo -e "  \\e[32m2.install component [..]\\e[0m"
echo -e "  \\e[32m3.uninstall component [..]\\e[0m"
echo -e "  \\e[32m4.start component [..]\\e[0m"
echo -e "  \\e[32m5.stop component [..]\\e[0m"
echo -e "  \\e[32m6.start download server [..]\\e[0m"
echo -e ""
echo -e "  \\e[32mq.quit\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m6\\e[0m,\\e[32mq\\e[0m(quit)]:"
}

## @description  check menu
## @audience     public
## @stability    stable
check_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}            DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu] > [prepare system environment]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.prepare operation system\\e[0m"
echo -e "  \\e[32m2.prepare operation system kernel\\e[0m"
echo -e "  \\e[32m3.prepare GCC version\\e[0m"
echo -e "  \\e[32m4.check GPU\\e[0m"
echo -e "  \\e[32m5.prepare user&group\\e[0m"
echo -e "  \\e[32m6.prepare nvidia environment\\e[0m"
echo -e ""
echo -e "  \\e[32mb.back main menu\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m5\\e[0m,\\e[32mb\\e[0m(back)]:"
}

## @description  install menu
## @audience     public
## @stability    stable
install_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}            DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu] > [install component]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.install etcd\\e[0m"
echo -e "  \\e[32m2.install docker\\e[0m"
echo -e "  \\e[32m3.install calico network\\e[0m"
echo -e "  \\e[32m4.install nvidia driver\\e[0m"
echo -e "  \\e[32m5.install nvidia docker\\e[0m"
echo -e "  \\e[32m6.install submarine autorun script\\e[0m"
echo -e "  \\e[32m7.install yarn\\e[0m"
echo -e ""
echo -e "  \\e[32mb.back main menu\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m7\\e[0m,\\e[32mb\\e[0m(back)]:"
}

## @description  unstall menu
## @audience     public
## @stability    stable
uninstall_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}            DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu] > [uninstall component]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.uninstall etcd\\e[0m"
echo -e "  \\e[32m2.uninstall docker\\e[0m"
echo -e "  \\e[32m3.uninstall calico network\\e[0m"
echo -e "  \\e[32m4.uninstall nvidia driver\\e[0m"
echo -e "  \\e[32m5.uninstall nvidia docker\\e[0m"
echo -e "  \\e[32m6.uninstall submarine autorun script\\e[0m"
echo -e "  \\e[32m7.uninstall yarn\\e[0m"
echo -e ""
echo -e "  \\e[32mb.back main menu\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m7\\e[0m,\\e[32mb\\e[0m(back)]:"
}

## @description  start menu
## @audience     public
## @stability    stable
start_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}            DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu] > [start component]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.start etcd\\e[0m"
echo -e "  \\e[32m2.start docker\\e[0m"
echo -e "  \\e[32m3.start calico network\\e[0m"
echo -e "  \\e[32m4.start yarn\\e[0m"
echo -e ""
echo -e "  \\e[32mb.back main menu\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m4\\e[0m,\\e[32mb\\e[0m(back)]:"
}

## @description  stop menu
## @audience     public
## @stability    stable
stop_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}    DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu] > [stop component]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.stop etcd\\e[0m"
echo -e "  \\e[32m2.stop docker\\e[0m"
echo -e "  \\e[32m3.stop calico network\\e[0m"
echo -e "  \\e[32m4.stop yarn\\e[0m"
echo -e ""
echo -e "  \\e[32mb.back main menu\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m4\\e[0m,\\e[32mb\\e[0m(back)]:"
}

## @description  install yarn menu
## @audience     public
## @stability    stable
install_yarn_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}            DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu] > [install yarn]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.instll YARM [every host]\\e[0m"
echo -e "  \\e[32m2.instll YARM container-executor [every host]\\e[0m"
echo -e "  \\e[32m3.instll YARM Timeline [one host]\\e[0m"
echo -e ""
echo -e "  \\e[32mb.back main menu\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m3\\e[0m,\\e[32mb\\e[0m(back)]:"
}

## @description  uninstall yarn menu
## @audience     public
## @stability    stable
uninstall_yarn_menu()
{
cat<<MENULIST
====================================================================================
                            SUBMARINE INSTALLER ${SUBMARINE_INSTALLER_VERSION}

HOST:${LOCAL_HOST_IP}            DOWNLOAD_SERVER:http://${DOWNLOAD_SERVER_IP}:${DOWNLOAD_SERVER_PORT}
====================================================================================
[Main menu] > [install yarn]
------------------------------------------------------------------------------------
MENULIST
echo -e "  \\e[32m1.uninstll Yarn\\e[0m"
echo -e ""
echo -e "  \\e[32mb.back main menu\\e[0m"
cat<<MENULIST
====================================================================================
MENULIST

echo -ne "Please input your choice [\\e[32m1\\e[0m-\\e[32m1\\e[0m,\\e[32mb\\e[0m(back)]:"
}

## @description  menu operation
## @audience     public
## @stability    stable
menu_index="0"
menu()
{
  clear
  # echo "menu_index-menu_choice=$menu_index-$menu_choice"
  case $menu_index in
    "0")
      main_menu
    ;;
    "1")
      check_menu
    ;;
    "2")
      install_menu
    ;;
    "3")
      uninstall_menu
    ;;
    "4")
      start_menu
    ;;
    "5")
      stop_menu
    ;;
    "6")
      start_download_server
    ;;
    "q")
      exit 1
    ;;
    *)
      echo "error input!"
      menu_index="0"
      menu_choice="0"
      main_menu
    ;;
  esac

  read -r menu_choice
}

## @description  menu process
## @audience     public
## @stability    stable
menu_process()
{
  process=0
  unset myselect
  # echo "debug=$menu_index-$menu_choice"
  case "$menu_index-$menu_choice" in
    "1-b"|"2-b"|"3-b"|"4-b"|"5-b"|"6-b"|"7-b"|"8-b")
      menu_index="0"
      menu_choice="0"
    ;;
# check system environment
    "1-1")
      myselect="y"
      check_operationSystem
    ;;
    "1-2")
      myselect="y"
      check_operationSystemKernel
    ;;
    "1-3")
      myselect="y"
      check_gccVersion
    ;;
    "1-4")
      myselect="y"
      check_GPU
    ;;
    "1-5")
      myselect="y"
      check_userGroup
    ;;
    "1-6")
      myselect="y"
      prepare_nvidia_environment
    ;;
# install component
    "2-1")
      echo -n "Do you want to install etcd?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        install_etcd
      fi
    ;;
    "2-2")
      echo -n "Do you want to install docker?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        install_docker
      fi
    ;;
    "2-3")
      echo -n "Do you want to install calico network?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        install_calico
      fi
    ;;
    "2-4")
      echo -n "Do you want to install nvidia driver?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        install_nvidia
      fi
    ;;
    "2-5")
      echo -n "Do you want to install nvidia docker?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        install_nvidia_docker
      fi
    ;;
    "2-6")
      echo -n "Do you want to install submarine auto start script?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        install_submarine
      fi
    ;;
    "2-7")
      echo -n "Do you want to install yarn?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        if [[ "${YARN_SECURITY}" = "true" ]];then
          install_yarn
        elif [[ "${YARN_SECURITY}" = "false" ]];then
          install_yarn_insecure
	else
	  echo "the YARN_SECURITY Value is not support !!!"
        fi
      fi
    ;;
# uninstall component
    "3-1")
      echo -n "Do you want to uninstall etcd?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        uninstall_etcd
      fi
    ;;
    "3-2")
      echo -n "Do you want to uninstall docker?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        uninstall_docker
      fi
    ;;
   "3-3")
      echo -n "Do you want to uninstall calico network?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        uninstall_calico
      fi
    ;;
    "3-4")
      echo -n "Do you want to uninstall nvidia driver?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        uninstall_nvidia
      fi
    ;;
    "3-5")
      echo -n "Do you want to uninstall nvidia docker?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        uninstall_nvidia_docker
      fi
    ;;
    "3-6")
      echo -n "Do you want to uninstall submarine autostart script?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        uninstall_submarine
      fi
    ;;
    "3-7")
      echo -n "Do you want to uninstall yarn?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        uninstall_yarn
      fi
    ;;
# startup component
    "4-1")
      echo -n "Do you want to startup etcd?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        start_etcd
      fi
    ;;
    "4-2")
      echo -n "Do you want to startup docker?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        start_docker
      fi
    ;;
    "4-3")
      echo -n "Do you want to startup calico network?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        start_calico
      fi
    ;;
    "4-4")
      echo -n "Do you want to startup yarn?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        start_yarn
      fi
    ;;
# stop component
    "5-1")
      echo -n "Do you want to stop etcd?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        stop_etcd
      fi
    ;;
    "5-2")
      echo -n "Do you want to stop docker?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        stop_docker
      fi
    ;;
    "5-3")
      echo -n "Do you want to stop calico network?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        stop_calico
      fi
    ;;
    "5-4")
      echo -n "Do you want to stop yarn?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        stop_yarn
      fi
    ;;
# install yarn
    "7-1")
      echo -n "Do you want to instll YARN?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
	if [[ "${YARN_SECURITY}" = "true" ]];then
          install_yarn
	elif [[ "${YARN_SECURITY}" = "false" ]];then
	  install_yarn_insecure
	else
	  echo "the YARN_SECURITY Value is not support !!!"
	fi
      fi
    ;;
    "7-2")
      echo -n "Do you want to instll container-executor?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        if [[ "${YARN_SECURITY}" = "true" ]];then
          install_yarn_container_executor
        elif [[ "${YARN_SECURITY}" = "false" ]];then
          install_yarn_container_executor_insecure
	else
	  echo "the YARN_SECURITY Value is not support !!!" 
        fi
      fi
    ;;
    "7-3")
      echo -n "Do you want to instll Timeline?[y|n]"
      read -r myselect
      if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
      then
        if [[ "${YARN_SECURITY}" = "true" ]];then
          install_timeline_server
        elif [[ "${YARN_SECURITY}" = "false" ]];then
          install_timeline_server_insecure
	else
           echo "the YARN_SECURITY Value is not support !!!"
        fi
      fi
    ;;
  esac

  if [[ "$myselect" = "y" || "$myselect" = "Y" ]]
  then
    process=1
  fi

#  echo "process=$process"
  return $process
}

