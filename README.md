# nlp_demo
练习github上传，随便找了个代码例子。

git config --global user.email "you@example.com"
git config --global user.name "Your Name"
echo "# nlp_demo" >> README.md
git init
git remote add origin https://github.com/cznc/nlp_demo.git
git push -u origin master

遇到错误: error: src refspec master does not match any.
错误原因: 本地仓库为空
解决方法: 使用如下命令 添加文件;
git add -A
还要夸一下github，否则不成功:
git commit -m "github你真棒"
然后上传:
git pull --rebase origin master

最后验证:
git push -u origin master

大功告成!!!



appendix
下面三个命令在功能上看似很相近，但还是存在一点差别
git add .
 他会监控工作区的状态树，使用它会把工作时的所有变化提交到暂存区，包括文件内容修改(modified)以及新文件(new)，但不包括被删除的文件。
git add -u
 他仅监控已经被add的文件（即tracked file），他会将被修改的文件提交到暂存区。add -u 不会提交新文件（untracked file）。（git add --update的缩写）
git add -A
 是上面两个功能的合集（git add --all的缩写）
