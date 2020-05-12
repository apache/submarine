module.exports = {
  presets: [
    '@vue/app',
    [
      '@babel/preset-env',
      {
        "corejs": "2.6.5",
        'useBuiltIns': 'entry'
      }
    ]
  ]
  // if your use import on Demand, Use this code
  // ,
  // plugins: [
  //   [ 'import', {
  //     'libraryName': 'ant-design-vue',
  //     'libraryDirectory': 'es',
  //     'style': true // `style: true` 会加载 less 文件
  //   } ]
  // ]
}
