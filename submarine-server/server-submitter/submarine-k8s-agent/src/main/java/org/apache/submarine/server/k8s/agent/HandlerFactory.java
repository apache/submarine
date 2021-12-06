package org.apache.submarine.server.k8s.agent;

import org.apache.submarine.server.k8s.agent.bean.CustomResourceType;
import org.apache.submarine.server.k8s.agent.handler.CustomResourceHandler;

public class HandlerFactory {

    private static String HANDLER_POSTFIX = "Handler";
    private static String HANDLER_PACKAGE = "org.apache.submarine.server.k8s.agent.handler";
    
    public static CustomResourceHandler getHandler(CustomResourceType crType) throws ClassNotFoundException, InstantiationException, IllegalAccessException {
        String handlerClassStr = HANDLER_PACKAGE + "." +  crType.getCustomResourceType() + HANDLER_POSTFIX;
        Class handlerClass = Class.forName(handlerClassStr);
        return (CustomResourceHandler)handlerClass.newInstance();
    }
}
