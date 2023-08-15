from quafu.users.userapi import User

user = User()
# user._load_account_token()
# user.save_apitoken()
print(user.api_token)
user.save_apitoken('hahah')
print(user.api_token)

# print(api)
# print('ha')
