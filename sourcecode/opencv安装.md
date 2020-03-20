build opencv:

参考网页：https://blog.csdn.net/CSDNNETVIP/article/details/77951990

1. Download opencv source code:<https://github.com/opencv/opencv/tree/4.1.0>
2. cd opencv; mkdir release; cd release;
3. cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DsCMAKE_INSTALL_PREFIX=/usr/local

4. make -j8

5. sudo make install

6. 设置环境变量：

   用vim打开/etc/ld.so.conf，注意要用sudo打开获得权限，不然无法修改，如：
   sudo vim /etc/ld.so.conf，在文件中加上一行
   /usr/loacal/lib，/user/loacal就是makefile中指定的安装路径

   再运行sudo ldconfig,

   修改bash.bashrc文件，sudo gedit /etc/bash.bashrc

   在文件末尾加入：

   PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
   export PKG_CONFIG_PATH

7. 查看opencv版本号：

pkg-config opencv --modversion

